#include <hos/sched/mem.hpp>
#include <hos/sched/pass.hpp>
#include <hos/util/fmt.hpp>
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
        MemStateList states;
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

using HierListFunc =
    std::function<std::vector<HierVertRef>(const HierVertRef &)>;
using SeqPred = std::function<bool(const SequenceRef &)>;

class SequenceDetector : public HierVertVisitor<bool> {
public:
    SequenceDetector(SeqPred inSet, HierListFunc getSuccs,
                     std::unordered_set<SequenceRef> &set,
                     std::vector<SequenceRef> &frontier,
                     std::vector<SequenceRef> &sink)
        : inSet(inSet),
          getSuccs(getSuccs),
          set(set),
          frontier(frontier),
          sink(sink) {}

    bool VisitSequence(const SequenceRef &seq) override {
        if (!inSet(seq)) return false;
        set.insert(seq);
        auto succs = getSuccs(seq);
        bool isFrontier = false, isSink = true;
        for (auto &succ : succs) {
            auto notIn = !Visit(succ);
            isFrontier |= notIn;
            isSink &= notIn;
        }
        if (isFrontier) AddUnique(frontier, seq);
        if (isSink) AddUnique(sink, seq);
        return true;
    }

    bool VisitInput(const HierInputRef &) override { return false; }
    bool VisitOutput(const HierOutputRef &) override { return false; }
    bool VisitGroup(const GroupRef &) override { return false; }

private:
    SeqPred inSet;
    HierListFunc getSuccs;
    std::unordered_set<SequenceRef> &set;
    std::vector<SequenceRef> &frontier;
    std::vector<SequenceRef> &sink;
};

static std::vector<ValueRef> gatherInputValues(
    const std::unordered_set<SequenceRef> set,
    const std::vector<SequenceRef> &inFront) {
    std::unordered_set<ValueRef> inputs;
    for (auto &in : inFront)
        for (auto &val : in->inputs) inputs.insert(val);
    for (auto &seq : set)
        for (auto &out : seq->outputs) inputs.erase(out);
    return std::vector(inputs.begin(), inputs.end());
}

static std::vector<ValueRef> gatherOutputValues(
    const std::vector<SequenceRef> &outFront) {
    std::vector<ValueRef> outputs;
    for (auto &out : outFront)
        for (auto &val : out->outputs) outputs.push_back(val);
    return outputs;
}

inline static void printSequence(const SequenceRef &seq) {
    fmt::print("{}\n", FmtList(
                           seq->ops, [](const OpRef &op) { return op->type; },
                           "", "", " "));
}

static void printGroup(const GroupRef &group) {
    fmt::print("GROUP\n");
    fmt::print("Input frontier:\n");
    for (auto &in : group->inFront) printSequence(in);
    fmt::print("Output frontier:\n");
    for (auto &out : group->outFront) printSequence(out);
    fmt::print("Entrance:\n");
    for (auto &entr : group->entrs) printSequence(entr);
    fmt::print("Exit:\n");
    for (auto &exit : group->exits) printSequence(exit);
    fmt::print("Input value:\n");
    for (auto &val : group->inputs) fmt::print("{}\n", val->name);
    fmt::print("Output value:\n");
    for (auto &val : group->outputs) fmt::print("{}\n", val->name);
    fmt::print("\n");
}

static GroupRef createGroup(const std::unordered_set<SequenceRef> &set,
                            const std::vector<SequenceRef> &inFront,
                            const std::vector<SequenceRef> &outFront,
                            const std::vector<SequenceRef> &entrs,
                            const std::vector<SequenceRef> &exits) {
    // Set fields of the group
    auto group = std::make_shared<Group>();
    group->seqs = std::vector(set.begin(), set.end());
    for (auto &seq : set) seq->group = group;
    group->inFront = inFront;
    group->outFront = outFront;
    group->inputs = gatherInputValues(set, inFront);
    group->outputs = gatherOutputValues(outFront);
    group->entrs = entrs;
    group->exits = exits;

    // Reconnnect vertices
    for (auto &entr : inFront) {
        entr->preds = Filter<decltype(entr->preds)>(
            entr->preds, [&](const HierVertWeakRef &predWeak) {
                auto pred = predWeak.lock();
                if (group->Contains(pred))
                    // keep this predecessor as it is in the group
                    return true;
                else {
                    // connect this predcessor to the group instead of the
                    // sequence
                    HierVertex::ReplaceSuccOfPred(pred, entr, group);
                    AddUnique(group->preds, predWeak);
                    return false;
                }
            });
    }

    for (auto &exit : outFront) {
        exit->succs = Filter<decltype(exit->succs)>(
            exit->succs, [&](const HierVertRef &succ) {
                if (group->Contains(succ))
                    return true;
                else {
                    HierVertex::ReplacePredOfSucc(succ, exit, group);
                    AddUnique(group->succs, succ);
                    return false;
                }
            });
    }

    return group;
}

inline static void makeGroupFromCell(const SequenceRef &cellOut) {
    std::unordered_set<SequenceRef> set;
    std::vector<SequenceRef> inFront, entrs;
    SequenceDetector(
        [&](const SequenceRef &seq) {
            return cellOut->PostDominates(*seq) &&
                   !seq->Dominates(*cellOut, true);
        },
        std::mem_fn(&HierVertex::Preds), set, inFront, entrs)
        .Visit(cellOut);
    createGroup(set, inFront, {cellOut}, entrs, {cellOut});
}

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

    // Find all cell outputs in reverse post-order
    std::vector<SequenceRef> cellOuts;
    for (auto v : RpoHierRange(hier)) {
        if (!Is<Sequence>(v)) continue;
        auto seq = Cast<Sequence>(v);
        if (isCellOut(seq)) cellOuts.push_back(seq);
    }

    // Build group from cells
    for (auto &out : cellOuts) makeGroupFromCell(out);
}

}  // namespace hos