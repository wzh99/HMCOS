#include <hos/sched/sched.hpp>
#include <hos/sched/mem.hpp>
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

struct Schedule {
    /// Sequence of ops as scheduling result
    std::vector<OpRef> result;
    /// Memory states of scheduled sequence
    MemStateVec states;
};

std::vector<OpRef> HierarchicalSchedule(const HierGraph &graph) {
    //
    return {};
}

}  // namespace hos