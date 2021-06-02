#include <hos/core/hier.hpp>

namespace hos {

HierGraph::HierGraph(const Graph &graph) {
    // Initialize inputs and outputs
    for (auto &in : graph.inputs) {
        auto hierIn = std::make_shared<HierInput>(in->value);
        this->inputs.push_back(hierIn);
        this->vertMap.insert({in, hierIn});
    }
    for (auto &out : graph.outputs) {
        auto hierOut = std::make_shared<HierOutput>(out->value);
        this->outputs.push_back(hierOut);
        this->vertMap.insert({out, hierOut});
    }

    // Map ops to sequences (with one op)
    std::vector<SequenceRef> seqs;
    for (auto &op : graph.ops) {
        auto seq = std::make_shared<Sequence>(op);
        seqs.push_back(seq);
        this->vertMap.insert({op, seq});
    }

    // Connect vertices
    for (auto &[op, seq] : this->vertMap) {
        for (auto &pred : op->preds)
            seq->preds.push_back(this->vertMap[pred.lock()]);
        for (auto &succ : op->succs) seq->succs.push_back(this->vertMap[succ]);
    }
}

}  // namespace hos