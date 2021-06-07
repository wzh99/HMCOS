#include <hos/core/hier.hpp>
#include <hos/util/viz.hpp>
#include <optional>

namespace hos {

Sequence::Sequence(const OpRef &op) : ops{op}, outputs(op->outputs) {
    for (auto &in : op->inputs) {
        if (in->kind == ValueKind::PARAM) continue;
        if (Contains(this->inputs, in)) continue;
        this->inputs.push_back(in);
    }
}

std::string Sequence::Format() const {
    return FmtList(
        ops, [](auto &op) { return op->type; }, "", "", "\n");
}

HierGraph::HierGraph(const Graph &graph) : graph(graph) {
    // Initialize inputs and outputs
    for (auto &in : graph.inputs) {
        auto hierIn = std::make_shared<HierInput>(in->value);
        inputs.push_back(hierIn);
        vertMap.insert({in, hierIn});
    }
    for (auto &out : graph.outputs) {
        auto hierOut = std::make_shared<HierOutput>(out->value);
        outputs.push_back(hierOut);
        vertMap.insert({out, hierOut});
    }

    // Map ops to sequences (with one op)
    std::vector<SequenceRef> seqs;
    for (auto &op : graph.ops) {
        auto seq = std::make_shared<Sequence>(op);
        seqs.push_back(seq);
        vertMap.insert({op, seq});
    }

    // Connect vertices
    for (auto &[op, seq] : vertMap) {
        for (auto &pred : op->preds) seq->preds.push_back(vertMap[pred.lock()]);
        for (auto &succ : op->succs) seq->succs.push_back(vertMap[succ]);
    }
}

class HierVizAllVisitor
    : public HierVertVisitor<Unit, DotCreator<VertexRef>::Context> {
public:
    using Context = DotCreator<VertexRef>::Context;

    HierVizAllVisitor(DotCreator<VertexRef> &creator) : creator(creator) {}

    void Visualize(const HierGraph &hier) {
        // Add nodes
        for (auto &out : hier.outputs) Visit(out, creator.Top());

        // Add edges
        for (auto &op : hier.graph.ops)
            for (auto &pred : op->preds) creator.Edge(pred.lock(), op);
        for (auto &out : hier.graph.outputs) {
            creator.Node(out, out->value->name);
            creator.Edge(out->Def(), out);
        }
    }

    Unit VisitInput(const HierInputRef &input, Context) override {
        auto &val = input->value;
        creator.Node(val->Vertex(), val->name);
        return {};
    }

    Unit VisitOutput(const HierOutputRef &output, Context ctx) override {
        for (auto &pred : output->preds) Visit(pred.lock(), ctx);
        return {};
    }

    Unit VisitSequence(const SequenceRef &seq, Context ctx) override {
        for (auto &pred : seq->preds) Visit(pred.lock(), ctx);
        auto cluster = ctx.Cluster();
        for (auto &op : seq->ops) cluster.Node(op, op->type);
        return {};
    }

    Unit VisitGroup(const GroupRef &group, Context ctx) override {
        for (auto &pred : group->preds) Visit(pred.lock(), ctx);
        auto cluster = ctx.Cluster();
        for (auto &exit : group->exits) Visit(exit, cluster);
        return {};
    }

private:
    DotCreator<VertexRef> &creator;
};

void HierGraph::VisualizeAll(const std::string &dir, const std::string &name,
                             const std::string &format) {
    DotCreator<VertexRef> creator(name);
    HierVizAllVisitor(creator).Visualize(*this);
    creator.Render(dir, format);
}

class HierVizTopVisitor : public HierVertVisitor<Unit> {
public:
    HierVizTopVisitor(DotCreator<HierVertRef> &creator) : creator(creator) {}

    void Visualize(const HierGraph &hier) {
        for (auto &in : hier.inputs) Visit(in);
    }

    Unit Visit(const HierVertRef &vert) override {
        if (Contains(memo, vert)) return {};
        creator.Node(vert, vert->Format());
        memo.insert({vert, {}});
        for (auto &succ : vert->succs) {
            Visit(succ);
            creator.Edge(vert, succ);
        }
        return {};
    }

    Unit VisitInput(const HierInputRef &input) override { return {}; }
    Unit VisitOutput(const HierOutputRef &output) override { return {}; }
    Unit VisitSequence(const SequenceRef &seq) override { return {}; }
    Unit VisitGroup(const GroupRef &group) override { return {}; }

private:
    DotCreator<HierVertRef> &creator;
};

void HierGraph::VisualizeTop(const std::string &dir, const std::string &name,
                             const std::string &format) {
    DotCreator<HierVertRef> creator(name);
    HierVizTopVisitor(creator).Visualize(*this);
    creator.Render(dir, format);
}

class HierDomVizVisitor
    : public DomTreeVisitor<HierVertex, Unit, const HierDomNodeRef &> {
public:
    HierDomVizVisitor(DotCreator<HierDomNodeRef> &creator) : creator(creator) {}

    Unit Visit(const HierDomNodeRef &node,
               const HierDomNodeRef &parent) override {
        creator.Node(node, node->vertex.lock()->Format());
        for (auto &childWeak : node->children) {
            auto child = childWeak.lock();
            Visit(child, node);
            creator.Edge(node, child);
        }
        return {};
    }

private:
    DotCreator<HierDomNodeRef> &creator;
};

void HierGraph::VisualizeDom(const std::string &dir, const std::string &name,
                             const std::string &format) {
    if (inputs.empty()) {
        LOG(ERROR) << "Input list of the hierarchical graph is empty.";
        return;
    }
    if (!inputs[0]->dom) {
        LOG(ERROR) << "Dominator tree has not been built.";
        return;
    }
    DotCreator<HierDomNodeRef> creator(name);
    HierDomVizVisitor(creator).Visit(inputs[0]->dom, nullptr);
    creator.Render(dir, format);
}

void HierGraph::VisualizePostDom(const std::string &dir,
                                 const std::string &name,
                                 const std::string &format) {
    if (outputs.empty()) {
        LOG(ERROR) << "Output list of the hierarchical graph is empty.";
        return;
    }
    if (!outputs[0]->postDom) {
        LOG(ERROR) << "Post-dominator tree has not been built.";
        return;
    }
    DotCreator<HierDomNodeRef> creator(name);
    HierDomVizVisitor(creator).Visit(outputs[0]->postDom, nullptr);
    creator.Render(dir, format);
}

}  // namespace hos