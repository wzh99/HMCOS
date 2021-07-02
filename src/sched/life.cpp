#include <hos/sched/life.hpp>
#include <hos/util/op.hpp>
#include <hos/util/viz.hpp>

namespace hos {

void LifetimeStat::Plot(const std::string &dir, const std::string &name,
                        std::optional<uint64_t> yMax,
                        const std::string &format) const {
    RectPlot plot(name);
    for (auto [t, m] : *this)
        plot.AddRect(float(t), 0.f, 1.f, float(m), "royalblue");
    if (yMax) plot.SetYMax(*yMax);
    plot.Render(dir, format);
}

std::pair<int32_t, uint64_t> UsageIter::operator*() {
    while (idx < values.size() && values[idx].gen == t) {
        alive.push_back(&values[idx]);
        sum += values[idx].value->type.Size();
        idx++;
    }
    RemoveIf(alive, [&](auto *block) {
        if (block->kill == t) {
            sum -= block->value->type.Size();
            return true;
        } else
            return false;
    });
    return {t, sum};
}

uint32_t OverlapInput(const OpRef &op) {
    // Cannot handle multiple output op
    if (op->outputs.size() > 1) return OVERLAP_FAILED;
    auto &out = op->outputs[0];

    // Check if it is element-wise
    if (!IsElementWise(op->type)) return OVERLAP_FAILED;

    // Output of op with single input can always overlap this input
    if (op->inputs.size() == 1) return 0;

    // Output of op with multiple outputs can only overlap input with same
    // type as it
    for (auto i = 0u; i < op->inputs.size(); i++) {
        auto &in = op->inputs[i];
        if (in->kind == ValueKind::PARAM) continue;
        if (in->type == out->type) return i;
    }

    return OVERLAP_FAILED;
}

LifetimeStat ComputeLifetime(const std::vector<OpRef> &opSeq,
                             const Graph &graph) {
    // Op sequence must be a full permutation of ops in graph
    LOG_ASSERT(opSeq.size() == graph.ops.size());

    // Initialize lifetime and use count of inputs
    std::unordered_map<ValueRef, Lifetime> valLife;
    std::unordered_map<ValueRef, uint32_t> useCnt;
    for (auto &in : graph.inputs) {
        auto &val = in->value;
        valLife.insert(
            {val, Lifetime{val, Lifetime::TIME_INPUT, Lifetime::TIME_UNKNOWN}});
        useCnt.insert({val, uint32_t(val->uses.size())});
    }

    // Compute lifetime
    for (auto i = 0; i < opSeq.size(); i++) {
        // Initialize lifetime of its outputs
        auto &op = opSeq[i];
        for (auto &out : op->outputs) {
            valLife.insert({out, Lifetime{out, i, Lifetime::TIME_UNKNOWN}});
            useCnt.insert({out, uint32_t(out->uses.size())});
        }

        // Compute lifetime ending of its inputs
        auto ovlIdx = OverlapInput(op);
        for (auto j = 0u; j < op->inputs.size(); j++) {
            auto &in = op->inputs[j];
            if (in->kind == ValueKind::PARAM) continue;
            if (!Contains(useCnt, in))
                LOG(FATAL) << fmt::format(
                    "Value {} used without definition before.", in->name);
            auto &cnt = useCnt[in];
            cnt--;
            // If output can overlap this input, its life ends before this op.
            // Otherwise, it must keep alive until computation of this op is
            // finished.
            if (cnt == 0) {
                valLife[in].kill = ovlIdx == j ? i : i + 1;
                useCnt.erase(in);
            }
        }
    }

    // Finalize lifetime of outputs
    int endTime = int32_t(opSeq.size());
    for (auto &out : graph.outputs) valLife[out->value].kill = endTime;

    // Sort lifetime
    auto blocks = Transform<std::vector<Lifetime>>(
        valLife, [](auto &p) { return p.second; });
    std::sort(blocks.begin(), blocks.end(), CmpByGenKill);

    return {{Lifetime::TIME_INPUT, endTime}, std::move(blocks)};
}

uint64_t EstimatePeak(const std::vector<OpRef> &seq,
                      const std::vector<InputRef> &inputs) {
    // Initialize use count and total memory size
    uint64_t total = 0;
    std::unordered_map<ValueRef, uint32_t> useCnt;
    for (auto &inVert : inputs) {
        auto inVal = inVert->value;
        useCnt.insert({inVal, uint32_t(inVal->uses.size())});
        total += inVal->type.Size();
    }

    // Estimate peak at each time
    uint64_t peak = total;
    std::vector<ValueRef> nextKill;  // values to be killed next time
    for (auto i = 0; i < seq.size(); i++) {
        // Generate outputs
        auto &op = seq[i];
        for (auto &out : op->outputs) {
            useCnt.insert({out, uint32_t(out->uses.size())});
            total += out->type.Size();
        }

        // Kill values that are left to this time
        for (auto &val : nextKill) total -= val->type.Size();
        nextKill.clear();

        // Scan inputs and possibly kill values that are no longer used
        auto ovlIdx = OverlapInput(op);
        for (auto j = 0u; j < op->inputs.size(); j++) {
            // Update use count of this input value
            auto &in = op->inputs[j];
            if (in->kind == ValueKind::PARAM) continue;
            if (!Contains(useCnt, in))
                LOG(FATAL) << fmt::format(
                    "Value {} used without definition before.", in->name);
            auto &cnt = useCnt[in];
            cnt--;

            // Choose time to kill this value
            if (cnt == 0) {
                if (ovlIdx == j)  // can overlap, kill this time
                    total -= in->type.Size();
                else  // live until end of this op, kill next time
                    nextKill.push_back(in);
                useCnt.erase(in);
            }
        }

        // Update peak memory
        peak = std::max(peak, total);
    }

    return peak;
}

}  // namespace hos