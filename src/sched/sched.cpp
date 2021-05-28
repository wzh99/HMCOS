#include <hos/sched/sched.hpp>
#include <hos/util/op.hpp>

namespace hos {

OpSeq ReversePostOrder(const Graph &graph) {
    std::vector<OpRef> seq;
    graph.Traverse([&](const VertexRef &v) {
        if (Is<Op>(v)) seq.push_back(As<Op>(v));
    });
    return seq;
}

class BruteForceSearcher {
public:
    BruteForceSearcher(const Graph &graph,
                       std::function<double(const OpSeq &)> metric,
                       std::function<void(const OpSeq &, double)> callback)
        : graph(graph), metric(metric), callback(callback) {}

    void Search() {
        // Initialize predecessor count
        std::unordered_map<OpRef, uint32_t> predCnt;
        for (auto &op : graph.ops)
            predCnt.insert({op, uint32_t(op->preds.size())});
        for (auto &input : graph.inputs)
            for (auto &succ : input->succs) predCnt[As<Op>(succ.lock())]--;

        // Begin searching
        best = INFINITY;
        OpSeq seq;
        search(seq, predCnt);
    }

private:
    void search(OpSeq &seq, std::unordered_map<OpRef, uint32_t> &predCnt) {
        // Call callbacks if sequence no worse than best is found
        if (predCnt.size() == 0) {
            auto m = this->metric(seq);
            if (m < best) {
                best = m;
                this->callback(seq, m);
            }
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
            for (auto &succWeak : op->succs) {
                auto succ = succWeak.lock();
                if (succ->GetKind() == Vertex::VertexKind::OP)
                    predCnt[As<Op>(succ)]--;
            }
            search(seq, predCnt);

            // Remove from sequence and restore predecessor count
            seq.pop_back();
            predCnt.insert({op, 0});
            for (auto &succWeak : op->succs) {
                auto succ = succWeak.lock();
                if (succ->GetKind() == Vertex::VertexKind::OP)
                    predCnt[As<Op>(succ)]++;
            }
        }
    }

    const Graph &graph;
    std::function<double(const OpSeq &)> metric;
    std::function<void(const OpSeq &, double)> callback;
    double best;
};

void BruteForceSearch(const Graph &graph,
                      std::function<double(const OpSeq &)> metric,
                      std::function<void(const OpSeq &, double)> callback) {
    BruteForceSearcher(graph, metric, callback).Search();
}

}  // namespace hos