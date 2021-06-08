#include <hos/sched/mem.hpp>
#include <hos/sched/pass.hpp>
#include <hos/util/op.hpp>

namespace hos {

class JoinVisitor : public HierVertVisitor<Unit> {
public:
    void Join(HierGraph &graph) {
        for (auto &in : graph.inputs) Visit(in);
    }

    Unit VisitInput(const HierInputRef &input) override {
        for (auto &succ : input->succs) Visit(succ);
        return {};
    }

    Unit VisitOutput(const HierOutputRef &output) override { return {}; }

    Unit VisitSequence(const SequenceRef &seq) override {
        // Check if all predecessors of the sequence have only one successor.
        // If so, it can be joint with any sequence which has lower transient
        // AND stable memory footprint. Otherwise, it can only be joint with
        // element-wise ops.
        auto canJoinAny =
            std::all_of(seq->preds.begin(), seq->preds.end(),
                        [](auto &v) { return v.lock()->succs.size() == 1; });

        // Initialize memory states
        auto cur = seq;
        MemStateSeq states;
        auto [inc, dec] = computeIncDec(cur->ops[0]);
        states.Append(inc, dec);

        // Iteratively join successors
        while (true) {
            // Cannot join if previous sequence has multiple successors or if
            // next has multiple predecessors
            if (cur->succs.size() != 1) break;
            if (!Is<Sequence>(cur->succs[0])) break;
            if (cur->succs[0]->preds.size() != 1) break;

            // Always join if next is an element-wise op
            auto next = Cast<Sequence>(cur->succs[0]);
            auto &nextOp = next->ops[0];
            auto isEw =
                OpTraitRegistry::Match(nextOp->type, OpTrait::ELEMENT_WISE);
            if (isEw) {
                join(cur, next);
                continue;
            }

            // Try join if next op is not element-wise
            if (!canJoinAny) break;
            auto [inc, dec] = computeIncDec(next->ops[0]);
            auto [trans, stable] = states.ComputeState(inc, dec);
            if (trans > states.Transients().Max() ||
                stable > states.Stables().Max())
                break;  // incurs higher footprint, stop here
            states.Append(inc, dec);
            join(cur, next);
        }

        return visitSuccs(seq);
    }

    Unit VisitGroup(const GroupRef &group) override {
        LOG(FATAL) << "Cannot run `JoinSequencePass` on a hierarchical graph "
                      "with groups.";
        return {};
    }

private:
    Unit visitSuccs(const SequenceRef &seq) {
        for (auto &succ : seq->succs) Visit(succ);
        return {};
    }

    static std::pair<uint64_t, uint64_t> computeIncDec(const OpRef &op) {
        auto inc = std::transform_reduce(
            op->outputs.begin(), op->outputs.end(), 0ull, std::plus<uint64_t>(),
            [](const ValueRef &val) { return val->type.Size(); });
        auto dec = std::transform_reduce(
            op->inputs.begin(), op->inputs.end(), 0ull, std::plus<uint64_t>(),
            [](const ValueRef &val) { return val->type.Size(); });
        return {inc, dec};
    }

    /// Join two sequences. The joint sequence will stored in `prev`, while
    /// `next` will be removed.
    void join(const SequenceRef &prev, const SequenceRef &next) {
        // Modify sequence data
        for (auto &op : next->ops) prev->ops.push_back(op);
        prev->outputs = next->outputs;

        // Reconnect vertices
        prev->succs = next->succs;
        next->preds.clear();
        HierVertex::Replace(next, prev);
    }
};

void JoinSequencePass::Run(HierGraph &hier) { JoinVisitor().Join(hier); }

void MakeGroupPass::Run(HierGraph &hier) {
    // Build dominator tree
    if (hier.inputs.empty()) {
        LOG(ERROR) << "Input list of the hierarchical graph is empty.";
        return;
    }
    if (hier.inputs.size() > 1)
        LOG(WARNING)
            << "Dominator tree will only be built for the first input vertex.";
    auto domNodes = DomBuilder<HierVertex>().Build(hier.inputs[0]);
    for (auto &node : domNodes) node->vertex.lock()->dom = node;

    // Build post-dominator tree
    if (hier.outputs.empty()) {
        LOG(ERROR) << "Output list of the hierarchical graph is empty.";
        return;
    }
    if (hier.outputs.size() > 1)
        LOG(WARNING) << "Post-dominator tree will only be built for the first "
                        "output vertex.";
    auto postDomNodes = DomBuilder<HierVertex>(std::mem_fn(&HierVertex::Succs),
                                               std::mem_fn(&HierVertex::Preds))
                            .Build(hier.outputs[0]);
    for (auto &node : postDomNodes) node->vertex.lock()->postDom = node;
}

}  // namespace hos