#include <hos/sched/sched.hpp>
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

OpSched::OpSched(const std::vector<OpRef> &opSeq, const Graph &graph)
    : opSeq(opSeq) {
    // Op sequence must be a full permutation of ops in graph
    LOG_ASSERT(opSeq.size() == graph.ops.size());

    // Initialize lifetime and use count of inputs
    std::unordered_map<ValueRef, uint32_t> useCnt;
    for (auto &in : graph.inputs) {
        auto &val = in->value;
        valLife.insert({val, Lifetime{-1, Lifetime::UNKNOWN}});
        useCnt.insert({val, val->uses.size()});
    }

    // Compute lifetime
    for (auto i = 0; i < opSeq.size(); i++) {
        // Initialize lifetime of its outputs
        auto &op = opSeq[i];
        for (auto &out : op->outputs) {
            valLife.insert({out, Lifetime{i, Lifetime::UNKNOWN}});
            useCnt.insert({out, out->uses.size()});
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
                valLife[in].end = ovlIdx == j ? i : i + 1;
                useCnt.erase(in);
            }
        }
    }

    // Finalize lifetime of outputs
    for (auto &out : graph.outputs) valLife[out->value].end = opSeq.size();
}

void hos::OpSched::Print() const {
    // Print operator sequence
    fmt::print("Operator sequence: \n");
    for (auto i = 0u; i < opSeq.size(); i++)
        fmt::print("{} {}\n", i, opSeq[i]->GetType());

    // Sort value lifetime
    auto vec = Transform<std::vector<std::pair<ValueRef, Lifetime> > >(
        valLife, [](auto &p) { return p; });
    std::sort(vec.begin(), vec.end(),
              [](auto &lhs, auto &rhs) { return lhs.second < rhs.second; });

    // Print lifetime
    fmt::print("\nValue lifetime: \n");
    for (auto &[val, lifetime] : vec)
        fmt::print("{}:{} {}\n", lifetime.begin, lifetime.end, val->name);
}

}  // namespace hos