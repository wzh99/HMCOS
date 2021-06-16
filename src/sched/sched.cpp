#include <hos/sched/life.hpp>
#include <hos/sched/mem.hpp>
#include <hos/sched/sched.hpp>

namespace hos {

std::vector<OpRef> ReversePostOrder(const Graph &graph) {
    std::vector<OpRef> seq;
    for (auto v : RpoVertRange(graph))
        if (Is<Op>(v)) seq.push_back(Cast<Op>(v));
    return seq;
}

class BruteForceSearcher {
public:
    BruteForceSearcher(
        const Graph &graph,
        std::function<uint64_t(const std::vector<OpRef> &)> metric,
        std::function<void(const std::vector<OpRef> &, uint64_t)> callback)
        : graph(graph), metric(metric), callback(callback) {}

    void Search() {
        // Initialize predecessor count
        std::unordered_map<OpRef, uint32_t> predCnt;
        for (auto &op : graph.ops)
            predCnt.insert({op, uint32_t(op->preds.size())});
        for (auto &input : graph.inputs)
            for (auto &succ : input->succs) predCnt[As<Op>(succ)]--;

        // Begin searching
        best = UINT64_MAX;
        std::vector<OpRef> seq;
        search(seq, predCnt);
    }

private:
    void search(std::vector<OpRef> &seq,
                std::unordered_map<OpRef, uint32_t> &predCnt) {
        // Prune sequences that are sub-optimal
        auto curMetric = this->metric(seq);
        if (curMetric >= best) return;

        // Call callback if a complete sequence is found
        if (predCnt.size() == 0) {
            best = curMetric;
            this->callback(seq, curMetric);
            return;
        }

        // Find candidate ops
        std::vector<OpRef> cand;
        for (auto &[op, cnt] : predCnt)
            if (cnt == 0) cand.push_back(op);

        // Choose one op and continue searching
        for (auto &op : cand) {
            // Add to sequence and update predecessor count
            seq.push_back(op);
            predCnt.erase(op);
            for (auto &succ : op->succs)
                if (succ->Kind() == VertexKind::OP) predCnt[As<Op>(succ)]--;
            search(seq, predCnt);

            // Remove from sequence and restore predecessor count
            seq.pop_back();
            predCnt.insert({op, 0});
            for (auto &succ : op->succs)
                if (succ->Kind() == VertexKind::OP) predCnt[As<Op>(succ)]++;

            // Prune if subsequence is already sub-optimal
            if (this->metric(seq) >= best) break;
        }
    }

    const Graph &graph;
    std::function<uint64_t(const std::vector<OpRef> &)> metric;
    std::function<void(const std::vector<OpRef> &, uint64_t)> callback;
    uint64_t best;
};

void BruteForceSearch(
    const Graph &graph,
    std::function<uint64_t(const std::vector<OpRef> &)> metric,
    std::function<void(const std::vector<OpRef> &, uint64_t)> callback) {
    BruteForceSearcher(graph, metric, callback).Search();
}

struct SchedResult {
    /// Scheduled sequence of ops
    std::vector<OpRef> seq;
    /// Memory states of scheduled sequence
    MemStateVec states;

    void Update(SchedResult &&other) {
        if (other.states.Peak() < this->states.Peak()) {
            this->seq.swap(other.seq);
            this->states.Swap(other.states);
        }
    }

    void Print() const {
        for (auto [op, state] : ZipRange(seq, states))
            LOG(INFO) << fmt::format("{:<18} {:>8}^ {:>8}_", op->type,
                                     state.first, state.second);
    }
};

template <class Vert>
struct PartialSchedResult : public SchedResult {
    /// Predecessor count of vertices
    /// This map serializes the graph structure to avoid traversal of the graph
    /// when computing zero-indegree sets.
    std::unordered_map<std::shared_ptr<Vert>, uint32_t> predCnt;
    /// Use count of values
    std::unordered_map<ValueRef, uint32_t> useCnt;
};

struct GroupContext {
    /// Group that this context describes
    GroupRef group;
    /// Whether each value consumed in this group are killed by it
    std::vector<int> kill;

    GroupContext(const GroupRef &group,
                 const std::unordered_map<ValueRef, uint32_t> &useCnt)
        : group(group),
          kill(Transform<decltype(kill)>(group->consumed, [&](auto &pair) {
              return pair.second == useCnt.at(pair.first);
          })) {}

    bool operator==(const GroupContext &other) const {
        return this->group == other.group && this->kill == other.kill;
    }
};

}  // namespace hos

namespace std {

template <>
struct hash<hos::GroupContext> {
    size_t operator()(const hos::GroupContext &ctx) const {
        return hos::Hash(ctx.group, ctx.kill);
    }
};

}  // namespace std

namespace hos {

template <class Vert>
inline static void decPredCount(
    const std::shared_ptr<Vert> &vert,
    std::unordered_map<std::shared_ptr<Vert>, uint32_t> &predCnt) {
    for (auto &succ : vert->succs) predCnt[succ]--;
}

/// Extract zero-indegree vertices from predecessor count map and move it to the
/// ordered vector
template <class Vert>
inline static void extractZeroIn(
    std::unordered_map<std::shared_ptr<Vert>, uint32_t> &predCnt,
    std::vector<std::shared_ptr<Vert>> &zeroIn) {
    for (auto &[vert, cnt] : predCnt)
        if (cnt == 0) Insert(zeroIn, vert);
    for (auto &vert : zeroIn) predCnt.erase(vert);
}

/// A sequence has only one possible schedule. This function also computes
/// memory states of each op and update predecessor count and use count map.
static SchedResult scheduleSequence(
    const SequenceRef &seq, std::unordered_map<ValueRef, uint32_t> &useCnt) {
    // Iterate each op and compute memory states
    MemStateVec states;
    for (auto &op : seq->ops) {
        // Consume use counts used by this value
        std::vector<ValueRef> killed;
        for (auto &val : op->inputs) {
            if (val->kind == ValueKind::PARAM) continue;
            auto cnt = --useCnt[val];
            if (cnt == 0) killed.push_back(val);
        }

        // See if output value can overlap one of the input
        auto ovlIdx = OverlapInput(op);
        if (ovlIdx != OVERLAP_FAILED && !Contains(killed, op->inputs[ovlIdx]))
            ovlIdx = OVERLAP_FAILED;

        // Compute increase in size at transition to transient state
        uint64_t inc = 0;
        if (ovlIdx == OVERLAP_FAILED)
            inc = std::transform_reduce(
                op->outputs.begin(), op->outputs.end(), 0ull, std::plus(),
                [](auto &val) { return val->type.Size(); });

        // Compute decrease in size at transition to stable state
        auto ovlVal = ovlIdx == OVERLAP_FAILED ? nullptr : op->inputs[0];
        auto dec = 0ull;
        for (auto &val : op->inputs) {
            if (val->kind == ValueKind::PARAM) continue;
            if (!Contains(killed, val)) continue;
            if (val == ovlVal) continue;
            dec += val->type.Size();
        }

        // Update memory states
        states.Append(inc, dec);

        // Remove killed values from use count map
        for (auto &val : killed) useCnt.erase(val);

        // Update use count for values generated by this op
        for (auto &val : op->outputs)
            useCnt.insert({val, uint32_t(val->uses.size())});
    }

    return {seq->ops, std::move(states)};
}

static SchedResult scheduleGroupRpo(
    const GroupRef &group, std::unordered_map<ValueRef, uint32_t> &useCnt) {
    // Initialize vertex range
    VertRange<HierVertex, RpoIter<HierVertex>> vertRange(
        Transform<std::vector<HierVertRef>>(
            group->exits, [](auto &exit) { return HierVertRef(exit); }));

    // Schedule each sequence in reverse post-order
    std::vector<OpRef> opSeq;
    MemStateVec states;
    for (auto &vert : vertRange) {
        auto seq = As<Sequence>(vert);
        auto [vertSeq, vertStates] = scheduleSequence(seq, useCnt);
        opSeq.insert(opSeq.end(), vertSeq.begin(), vertSeq.end());
        states.Extend(vertStates);
    }

    return {std::move(opSeq), std::move(states)};
}

class HierScheduler {
public:
    HierScheduler(const HierGraph &hier) : hier(hier) {}

    std::vector<OpRef> Schedule() {
        // Initialize predecessor count for vertices
        std::unordered_map<HierVertRef, uint32_t> predCnt;
        for (auto vert : RpoHierRange(hier)) {
            if (Is<HierInput>(vert) || Is<HierOutput>(vert)) continue;
            predCnt.insert({vert, uint32_t(vert->preds.size())});
        }
        auto nVert = predCnt.size();

        // Initialize use count of values
        std::unordered_map<ValueRef, uint32_t> useCnt;
        for (auto &input : hier.inputs) {
            for (auto &succ : input->succs) predCnt[succ]--;
            auto &val = input->value;
            useCnt.insert({val, uint32_t(val->uses.size())});
        }

        // Initialize partial result
        std::vector<HierVertRef> zeroIn;
        extractZeroIn(predCnt, zeroIn);
        auto initSize = std::transform_reduce(
            hier.inputs.begin(), hier.inputs.end(), 0ull, std::plus(),
            [](auto &input) { return input->value->type.Size(); });
        std::unordered_map<std::vector<HierVertRef>,
                           PartialSchedResult<HierVertex>>
            memo;
        memo.insert({zeroIn, {{{}, MemStateVec(initSize)}, predCnt, useCnt}});

        // Iterate each step
        for (auto i = 0u; i < nVert; i++) {
            // Iterate each partial result and build partial schedule with one
            // more vertex
            decltype(memo) newMemo;
            for (const auto &[zeroIn, result] : memo) {
                // Try add another vertex to the sequence
                for (auto &vert : zeroIn) {
                    // Schedule this vertex
                    auto useCnt = result.useCnt;
                    auto [vertSeq, vertStates] =
                        scheduleVertex(vert, useCnt, result.states);

                    // Update result of this partial schedule
                    auto seq = result.seq;  // extend op sequence
                    seq.insert(seq.end(), vertSeq.begin(), vertSeq.end());
                    auto states = result.states;  // extend memory states
                    states.Extend(vertStates);
                    auto predCnt = result.predCnt;  // update zero-indegree set
                    decPredCount(vert, predCnt);
                    auto newZeroIn = zeroIn;
                    Remove(newZeroIn, vert);
                    extractZeroIn(predCnt, newZeroIn);
                    PartialSchedResult<HierVertex> newResult{
                        {seq, states}, std::move(predCnt), std::move(useCnt)};

                    // Memoize this result
                    if (Contains(newMemo, newZeroIn))
                        newMemo[newZeroIn].Update(std::move(newResult));
                    else
                        newMemo.insert({newZeroIn, std::move(newResult)});
                }
            }
            newMemo.swap(memo);
        }

        return memo[{}].seq;
    }

private:
    SchedResult scheduleVertex(const HierVertRef &vert,
                               std::unordered_map<ValueRef, uint32_t> &useCnt,
                               const MemStateVec &prevStates) {
        switch (vert->Kind()) {
            case HierKind::SEQUENCE:
                return scheduleSequence(Cast<Sequence>(vert), useCnt);

            case HierKind::GROUP: {
                // Check if there is memoized result
                auto group = Cast<Group>(vert);
                GroupContext ctx(group, useCnt);
                if (Contains(groupMemo, ctx)) return groupMemo[ctx];

                // Try schedule using reverse post-order
                auto newUseCnt = useCnt;
                auto rpoResult = scheduleGroupRpo(group, newUseCnt);

                // Use RPO schedule if peak is not lifted
                if (rpoResult.states.Peak() + prevStates.Latest() <=
                    prevStates.Peak())
                    return rpoResult;

                //
            }
        }
        LOG(FATAL) << "Unreachable.";
    }

    /// Hierarchical graph to be scheduled
    const HierGraph &hier;
    /// Scheduling result of each group, under different contexts
    std::unordered_map<GroupContext, SchedResult> groupMemo;
};

std::vector<OpRef> HierarchicalSchedule(const HierGraph &hier) {
    return HierScheduler(hier).Schedule();
}

}  // namespace hos