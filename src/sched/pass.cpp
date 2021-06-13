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
        MemStateVec states;
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
        HierVertex::ReplacePredOfAllSuccs(next, prev);
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
    LOG(INFO) << FmtList(
        seq->ops, [](const OpRef &op) { return op->type; }, "", "", " ");
}

static void printGroup(const GroupRef &group) {
    LOG(INFO) << "# GROUP";
    LOG(INFO) << "## Input frontier:";
    for (auto &in : group->inFront) printSequence(in);
    LOG(INFO) << "## Output frontier:";
    for (auto &out : group->outFront) printSequence(out);
    LOG(INFO) << "## Entrance:";
    for (auto &entr : group->entrs) printSequence(entr);
    LOG(INFO) << "## Exit:";
    for (auto &exit : group->exits) printSequence(exit);
    LOG(INFO) << "## Value consumed:";
    for (auto &[val, cnt] : group->consumed)
        LOG(INFO) << val->name << " " << cnt;
    LOG(INFO) << "## Value produced:";
    for (auto &[val, cnt] : group->produced)
        LOG(INFO) << val->name << " " << cnt;
}

inline static void initOrInc(std::unordered_map<ValueRef, uint32_t> &map,
                             const ValueRef &key) {
    if (Contains(map, key))
        map[key]++;
    else
        map.insert({key, 1});
}

static std::vector<std::pair<ValueRef, uint32_t>> countConsumed(
    const std::unordered_set<SequenceRef> &set,
    const std::vector<SequenceRef> &inFront) {
    // Find all values consumed by input frontiers
    std::unordered_map<ValueRef, uint32_t> consumed;
    for (auto &seq : inFront)
        for (auto &in : seq->inputs) initOrInc(consumed, in);

    // Remove count of those produced by sequences in the set
    for (auto &seq : set)
        for (auto &out : seq->outputs)
            if (Contains(consumed, out)) consumed[out]--;

    // Prune pairs whose count is zero
    std::vector<std::pair<ValueRef, uint32_t>> vec;
    for (auto &[val, cnt] : consumed)
        if (cnt != 0) vec.push_back({val, cnt});

    return vec;
}

static std::vector<std::pair<ValueRef, uint32_t>> countProduced(
    const std::unordered_set<SequenceRef> &set,
    const std::vector<SequenceRef> &outFront) {
    // Find all values produced by output frontiers
    std::unordered_map<ValueRef, uint32_t> produced;
    for (auto &seq : outFront)
        for (auto &out : seq->outputs)
            produced.insert({out, uint32_t(out->uses.size())});

    // Remove count of those consumed by sequences in the set
    for (auto &seq : set)
        for (auto &in : seq->inputs)
            if (Contains(produced, in)) produced[in]--;

    // Prune pairs whose count is zero
    std::vector<std::pair<ValueRef, uint32_t>> vec;
    for (auto &[val, cnt] : produced)
        if (cnt != 0) vec.push_back({val, cnt});

    return vec;
}

static GroupRef createGroup(const std::unordered_set<SequenceRef> &set,
                            const std::vector<SequenceRef> &inFront,
                            const std::vector<SequenceRef> &outFront,
                            const std::vector<SequenceRef> &entrs,
                            const std::vector<SequenceRef> &exits) {
    // Set fields of the group
    auto group = std::make_shared<Group>();
    for (auto &seq : set) seq->group = group;
    group->inFront = inFront;
    group->outFront = outFront;
    group->consumed = countConsumed(set, inFront);
    group->produced = countProduced(set, outFront);
    group->entrs = entrs;
    group->exits = exits;

    // Reconnnect vertices
    for (auto &front : inFront) {
        front->preds = Filter<decltype(front->preds)>(
            front->preds, [&](const HierVertWeakRef &predWeak) {
                auto pred = predWeak.lock();
                if (group->Contains(pred))
                    // keep this predecessor as it is in the group
                    return true;
                else {
                    // connect this predcessor to the group instead of the
                    // sequence
                    HierVertex::ReplaceSuccOfPred(pred, front, group);
                    AddUnique(group->preds, predWeak);
                    return false;
                }
            });
    }

    for (auto &front : outFront) {
        front->succs = Filter<decltype(front->succs)>(
            front->succs, [&](const HierVertRef &succ) {
                if (group->Contains(succ))
                    return true;
                else {
                    HierVertex::ReplacePredOfSucc(succ, front, group);
                    AddUnique(group->succs, succ);
                    return false;
                }
            });
    }
    
    return group;
}

class OutputSizeOptimizer {
public:
    OutputSizeOptimizer(const std::unordered_set<SequenceRef> &allSeqs,
                        const SequenceRef &root)
        : allSeqs(allSeqs), root(root) {}

    std::vector<SequenceRef> Optimize() {
        // Build predecessor count map
        std::unordered_map<SequenceRef, uint32_t> predCount;
        for (auto &seq : allSeqs)
            predCount.insert({seq, uint32_t(seq->preds.size())});
        predCount[root] = 0;

        // Begin searching all possible intrusion on the cell
        std::vector<SequenceRef> chosen;
        std::unordered_map<SequenceRef, uint32_t> succCount;
        minSize = UINT64_MAX;
        search(chosen, predCount, succCount);

        return bestSet;
    }

private:
    void search(std::vector<SequenceRef> &chosen,
                std::unordered_map<SequenceRef, uint32_t> &predCount,
                std::unordered_map<SequenceRef, uint32_t> &succCount) {
        // Check if this set has been searched before
        if (Contains(memo, chosen)) return;

        // Compute size of output frontier
        uint64_t size = 0;
        for (auto &seq : chosen) {
            if (succCount[seq] == 0) continue;
            size += std::transform_reduce(
                seq->outputs.begin(), seq->outputs.end(), 0ull, std::plus(),
                [](const ValueRef &val) { return val->type.Size(); });
        }

        if (size != 0) {
            memo.insert({chosen, size});
            if (size < minSize ||
                (size == minSize && chosen.size() > bestSet.size())) {
                minSize = size;
                bestSet = chosen;
            }
        }

        // Find all zero-predecessor sequences
        std::vector<SequenceRef> cand;
        for (auto &[seq, count] : predCount)
            if (count == 0) cand.push_back(seq);

        // Choose one sequence and search further
        for (auto &seq : cand) {
            // Add to set
            auto pos = std::upper_bound(chosen.begin(), chosen.end(), seq) -
                       chosen.begin();
            chosen.insert(chosen.begin() + pos, seq);

            // Update predecessor and successor count
            predCount.erase(seq);
            for (auto &succ : filterSeqs(seq->succs)) predCount[succ]--;
            succCount.insert({seq, uint32_t(seq->succs.size())});
            for (auto &pred : filterSeqs(seq->Preds())) succCount[pred]--;

            // Search next sequence
            search(chosen, predCount, succCount);

            // Remove from set
            chosen.erase(chosen.begin() + pos);

            // Restore predecessor and successor count
            predCount.insert({seq, 0});
            for (auto &succ : filterSeqs(seq->succs)) predCount[succ]++;
            succCount.erase(seq);
            for (auto &pred : filterSeqs(seq->Preds())) succCount[pred]++;
        }
    }

    std::vector<SequenceRef> filterSeqs(const std::vector<HierVertRef> &verts) {
        std::vector<SequenceRef> seqs;
        for (auto &v : verts) {
            if (!Is<Sequence>(v)) continue;
            auto seq = Cast<Sequence>(v);
            if (!Contains(allSeqs, seq)) continue;
            seqs.push_back(seq);
        }
        return seqs;
    }

    const std::unordered_set<SequenceRef> &allSeqs;
    const SequenceRef &root;
    std::unordered_map<std::vector<SequenceRef>, uint64_t> memo;
    std::vector<SequenceRef> bestSet;
    uint64_t minSize;
};

inline static void makeGroupFromCell(const SequenceRef &cellOut) {
    // Detect input frontier of the group
    std::unordered_set<SequenceRef> seqs;
    std::vector<SequenceRef> inFront, entrs;
    SequenceDetector(
        [&](const SequenceRef &seq) {
            return cellOut->PostDominates(*seq) &&
                   !seq->Dominates(*cellOut, true);
        },
        std::mem_fn(&HierVertex::Preds), seqs, inFront, entrs)
        .Visit(cellOut);

    // Detect output frontier of the group by intruding on other cells
    std::unordered_set<SequenceRef> intruded;
    std::vector<SequenceRef> outFront, exits;
    SequenceDetector(
        [&](const SequenceRef &seq) { return cellOut->Dominates(*seq); },
        std::mem_fn(&HierVertex::Succs), intruded, outFront, exits)
        .Visit(cellOut);

    // Directly create group if intrusion is not possible
    if (Contains(outFront, cellOut)) {
        createGroup(seqs, inFront, {cellOut}, entrs, {cellOut});
        return;
    }

    // Try choosing a subset of intruded sequences that minimize their output
    // sizes
    auto minSizeSet = OutputSizeOptimizer(intruded, cellOut).Optimize();

    // Build group with intruded sequences
    outFront.clear();
    exits.clear();
    SequenceDetector(
        [&](const SequenceRef &seq) { return Contains(minSizeSet, seq); },
        std::mem_fn(&HierVertex::Succs), seqs, outFront, exits)
        .Visit(cellOut);
    createGroup(seqs, inFront, outFront, entrs, exits);
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
    for (auto &out : cellOuts) {
        if (out->group.lock()) continue;
        makeGroupFromCell(out);
    }
}

}  // namespace hos