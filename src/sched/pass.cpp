#include <hos/sched/mem.hpp>
#include <hos/sched/pass.hpp>
#include <hos/util/fmt.hpp>
#include <hos/util/op.hpp>

namespace hos {

class JoinVisitor : public HierVertVisitor<Unit> {
public:
    JoinVisitor(HierGraph &hier) : hier(hier) {}

    void Join() {
        for (auto &in : hier.inputs) Visit(in);
    }

    Unit VisitInput(const HierInputRef &input) override {
        for (auto &succ : input->succs) Visit(succ);
        return {};
    }

    Unit VisitOutput(const HierOutputRef &output) override { return {}; }

    Unit VisitSequence(const SequenceRef &seq) override {
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
            auto next = Cast<Sequence>(cur->succs[0]);
            if (next->preds.size() != 1) break;

            // Try join if next op is not element-wise
            auto [inc, dec] = computeIncDec(next->ops[0]);
            auto [s, t] = states.ComputeState(inc, dec);
            if (s > states.Stables().Max() || t > states.Latest())
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
        std::vector<ValueRef> killed;
        for (auto &in : op->inputs)
            if (std::all_of(in->uses.begin(), in->uses.end(),
                            [&](auto &use) { return use.lock() == op; }))
                AddUnique(killed, in);
        return ComputeIncDec(op, killed);
    }

    /// Join two sequences. The joint sequence will stored in `prev`, while
    /// `next` will be removed.
    void join(const SequenceRef &prev, const SequenceRef &next) {
        // Modify sequence data
        for (auto &op : next->ops) {
            prev->ops.push_back(op);
            hier.opToSeq[op] = prev;
        }
        prev->outputs = next->outputs;

        // Reconnect vertices
        prev->succs = next->succs;
        HierVertex::ReplacePredOfAllSuccs(next, prev);
    }

    HierGraph &hier;
};

void JoinSequencePass::Run(HierGraph &hier) { JoinVisitor(hier).Join(); }

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
    for (auto &seq : inFront) {
        for (auto &in : seq->inputs) {
            auto def = in->def.lock();
            if (std::any_of(set.begin(), set.end(),
                            [&](auto &seq) { return seq->Contains(def); }))
                continue;
            initOrInc(consumed, in);
        }
    }

    return {consumed.begin(), consumed.end()};
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
    // Create group object
    auto group = std::make_shared<Group>();

    // Set fields of sequences
    for (auto &seq : set) seq->group = group;

    // Set fields of the group
    group->seqs = std::vector(set.begin(), set.end());
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
                if (group->Contains<Sequence>(pred))
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
                if (group->Contains<Sequence>(succ))
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

/// Use DP to find a subset of intruded sequences which minimize size of its
/// outputs.
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
            auto idx = Insert(chosen, seq);

            // Update predecessor and successor count
            predCount.erase(seq);
            for (auto &succ : filterSeqs(seq->succs)) predCount[succ]--;
            succCount.insert({seq, uint32_t(seq->succs.size())});
            for (auto &pred : filterSeqs(seq->Preds())) succCount[pred]--;

            // Search next sequence
            search(chosen, predCount, succCount);

            // Remove from set
            chosen.erase(chosen.begin() + idx);

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

bool MakeGroupPass::intrusion = true;

std::function<bool(const SequenceRef &)> MakeGroupPass::isCellOut =
    [](auto &seq) { return seq->ops.front()->type == "Concat"; };

inline static void makeGroupFromCell(const SequenceRef &cellOut) {
    // Detect input frontier of the group
    std::unordered_set<SequenceRef> seqs;
    std::vector<SequenceRef> cellInFront, cellEntrs;
    SequenceDetector(
        [&](const SequenceRef &seq) { return cellOut->PostDominates(*seq); },
        std::mem_fn(&HierVertex::Preds), seqs, cellInFront, cellEntrs)
        .Visit(cellOut);

    // Detect output frontier of the group by intruding on other cells
    std::unordered_set<SequenceRef> intruded;
    std::vector<SequenceRef> intrOutFront, intrExits;
    SequenceDetector(
        [&](const SequenceRef &seq) { return cellOut->Dominates(*seq); },
        std::mem_fn(&HierVertex::Succs), intruded, intrOutFront, intrExits)
        .Visit(cellOut);

    // Directly create group if intrusion is not required or not possible
    if (!MakeGroupPass::intrusion || Contains(intrOutFront, cellOut)) {
        createGroup(seqs, cellInFront, {cellOut}, cellEntrs, {cellOut});
        return;
    }

    // Try choosing a subset of intruded sequences that minimize their output
    // sizes
    auto minSizeSet = OutputSizeOptimizer(intruded, cellOut).Optimize();
    if (minSizeSet.size() <= 2) {  // don't intrude if the subset is trivial
        createGroup(seqs, cellInFront, {cellOut}, cellEntrs, {cellOut});
        return;
    }

    // Find output frontier and exits of intruded sequences
    intruded.clear();
    intrOutFront.clear();
    intrExits.clear();
    SequenceDetector(
        [&](const SequenceRef &seq) { return Contains(minSizeSet, seq); },
        std::mem_fn(&HierVertex::Succs), intruded, intrOutFront, intrExits)
        .Visit(cellOut);
    intruded.erase(cellOut);

    // Find input frontier and entrance of intruded sequences
    std::vector<SequenceRef> intrInFront, intrEntrs;
    for (auto &succ : cellOut->succs) {
        if (!Is<Sequence>(succ)) continue;
        auto seq = Cast<Sequence>(succ);
        intrInFront.push_back(seq);
        if (!std::any_of(seq->preds.begin(), seq->preds.end(), [&](auto &pred) {
                return Is<Sequence>(pred.lock()) &&
                       Contains(intruded, Cast<Sequence>(pred.lock()));
            }))
            intrEntrs.push_back(seq);
    }

    // Create cell group and intruded group
    createGroup(seqs, cellInFront, {cellOut}, cellEntrs, {cellOut});
    createGroup(intruded, intrInFront, intrOutFront, intrEntrs, intrExits);
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

    // Find all cell outputs in reverse post-order, also backup predecessors and
    // successors
    std::vector<SequenceRef> cellOuts;
    for (auto v : RpoHierRange(hier)) {
        v->BackupEdges();
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