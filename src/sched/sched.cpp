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

}  // namespace hos