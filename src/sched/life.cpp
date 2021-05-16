#include <hos/sched/life.hpp>
#include <hos/util/op.hpp>

namespace hos {

static constexpr uint32_t OVERLAP_FAILED = UINT32_MAX;

/// Whether the only output of this op can overlap one of the input
static uint32_t overlapInput(const OpRef &op) {
    // Cannot multiple output op
    if (op->outputs.size() > 1) return OVERLAP_FAILED;
    auto &out = op->outputs[0];

    // Check if it is element-wise
    if (OpTraitRegistry::Match(op->GetType(), OpTrait::ELEMENT_WISE)) {
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

    return OVERLAP_FAILED;
}

std::vector<Lifetime> ComputeLifetime(const OpSeq &opSeq, const Graph &graph) {
    // Op sequence must be a full permutation of ops in graph
    LOG_ASSERT(opSeq.size() == graph.ops.size());

    // Initialize lifetime and use count of inputs
    std::unordered_map<ValueRef, Lifetime> valLife;
    std::unordered_map<ValueRef, uint32_t> useCnt;
    for (auto &in : graph.inputs) {
        auto &val = in->value;
        valLife.insert({val, Lifetime{val, -1, Lifetime::UNKNOWN}});
        useCnt.insert({val, uint32_t(val->uses.size())});
    }

    // Compute lifetime
    for (auto i = 0; i < opSeq.size(); i++) {
        // Initialize lifetime of its outputs
        auto &op = opSeq[i];
        for (auto &out : op->outputs) {
            valLife.insert({out, Lifetime{out, i, Lifetime::UNKNOWN}});
            useCnt.insert({out, uint32_t(out->uses.size())});
        }

        // Compute lifetime ending of its inputs
        auto ovlIdx = overlapInput(op);
        for (auto j = 0u; j < op->inputs.size(); j++) {
            auto &in = op->inputs[j];
            if (in->kind == ValueKind::PARAM) continue;
            if (!Contains(useCnt, in))
                LOG(FATAL) << "Operator sequence is not a topological sorting "
                              "of the computation graph.";
            auto &cnt = useCnt[in];
            cnt--;
            // If output can overlap this input, its life ends before this op.
            // Otherwise, it must keep alive until computation of this op
            // finishes.
            if (cnt == 0) {
                valLife[in].kill = ovlIdx == j ? i : i + 1;
                useCnt.erase(in);
            }
        }
    }

    // Finalize lifetime of outputs
    for (auto &out : graph.outputs)
        valLife[out->value].kill = int32_t(opSeq.size());

    // Sort lifetime
    auto ltVec = Transform<std::vector<Lifetime> >(
        valLife, [](auto &p) { return p.second; });
    std::sort(ltVec.begin(), ltVec.end(), CmpByGenKill());

    return ltVec;
}

}  // namespace hos