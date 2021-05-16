#include <hos/sched/order.hpp>

namespace hos {

OpSched ReversePostOrder(const Graph &graph) {
    std::vector<OpRef> seq;
    graph.Traverse([&](const VertexRef &v) {
        if (Is<Op>(v)) seq.push_back(As<Op>(v));
    });
    return OpSched(seq, graph);
}

}  // namespace hos