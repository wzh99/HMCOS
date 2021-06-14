#include <hos/sched/life.hpp>
#include <hos/sched/mem.hpp>
#include <hos/sched/sched.hpp>
#include <hos/util/op.hpp>

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
    std::vector<OpRef> sched;
    /// Memory states of scheduled sequence
    MemStateVec states;
};

/// Indicates ownerwship values in each group
enum class ValueOwnership {
    /// A group owns a value if it consumes all the remaining use counts of the
    /// value during scheduling.
    OWN,
    /// A group borrows a value if there are still remaining use counts after
    /// scheduling.
    BORROW,
};

struct GroupContext {
    /// Group that this context describes
    GroupRef group;
    /// Ownership of each value consumed in this group, it must match `consumed`
    /// field of the group.
    std::vector<ValueOwnership> own;
};

}  // namespace hos

namespace std {

template <>
struct hash<hos::GroupContext> {
    size_t operator()(const hos::GroupContext &ctx) const {
        return hos::Hash(ctx.group, ctx.own);
    }
};

}  // namespace std

namespace hos {

class HierScheduler {
public:
    HierScheduler(const HierGraph &hier) : hier(hier) {}

    std::vector<OpRef> Schedule() {
        // Compute upper bound of memory peak
        best = EstimatePeak(ReversePostOrder(hier.graph), hier.graph.inputs);

        // Initialize predecessor count for vertices
        std::unordered_map<HierVertRef, uint32_t> predCnt;
        for (auto v : RpoHierRange(hier)) {
            if (Is<HierInput>(v) || Is<HierOutput>(v)) continue;
            predCnt.insert({v, uint32_t(v->preds.size())});
        }

        // Initialize use count for values
        std::unordered_map<ValueRef, uint32_t> useCnt;
        for (auto &input : hier.inputs) {
            for (auto &succ : input->succs) predCnt[succ]--;
            auto &val = input->value;
            useCnt.insert({val, uint32_t(val->uses.size())});
        }

        return {};
    }

private:
    /// Hierarchical graph to be scheduled
    const HierGraph &hier;
    /// Partial result of hierarchical graph scheduling
    std::unordered_map<std::vector<GroupRef>, SchedResult> hierSched;
    /// Scheduling result of each group, under different contexts
    std::unordered_map<GroupContext, SchedResult> groupSched;
    /// Minimal memory peak up to now
    uint64_t best;
};

std::vector<OpRef> HierarchicalSchedule(const HierGraph &hier) {
    return HierScheduler(hier).Schedule();
}

}  // namespace hos