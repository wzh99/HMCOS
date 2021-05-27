#include <hos/core/graph.hpp>
#include <hos/util/fmt.hpp>
#include <hos/util/viz.hpp>
#include <unordered_map>

namespace hos {

Graph::Graph(const onnx::ModelProto &model, const std::string &name) {
    // Create name of this graph
    auto &graph = model.graph();
    this->name = name.size() == 0 ? graph.name() : name;

    // Build name-value map
    std::unordered_map<std::string, ValueRef> nameToVal;
    // Inputs
    for (auto &info : graph.input()) {
        auto val = std::make_shared<Value>(Value::CreateInput(info));
        auto in = std::make_shared<Input>(val);
        val->input = in;
        inputs.push_back(in);
        nameToVal.insert({info.name(), val});
    }
    // Outputs
    for (auto &info : graph.output()) {
        auto val = std::make_shared<Value>(Value::CreateResult(info));
        outputs.push_back(std::make_shared<Output>(val));
        nameToVal.insert({info.name(), val});
    }
    // Parameters
    for (auto &tensor : graph.initializer()) {
        auto val = std::make_shared<Value>(Value::CreateParam(tensor));
        params.push_back(val);
        nameToVal.insert({tensor.name(), val});
    }
    // Intermediates
    for (auto &info : graph.value_info()) {
        auto val = std::make_shared<Value>(Value::CreateResult(info));
        nameToVal.insert({info.name(), val});
    }

    // Build ops
    for (auto &node : graph.node()) {
        auto op = std::make_shared<Op>(&node);
        // Input values
        for (auto &in : node.input()) {
            if (!Contains(nameToVal, in))
                LOG(FATAL) << fmt::format(
                    "Cannot find information of value {}.", in);
            auto &inVal = nameToVal[in];
            op->inputs.push_back(inVal);
            inVal->uses.push_back(op);
        }
        // Output values
        for (auto &out : node.output()) {
            if (!Contains(nameToVal, out))
                LOG(FATAL) << fmt::format(
                    "Cannot find information of value {}.", out);
            auto &outVal = nameToVal[out];
            op->outputs.push_back(outVal);
            outVal->def = op;
        }
        ops.push_back(op);
    }

    // Connect vertices
    ConnectVerts();
}

void Graph::ConnectVerts() {
    for (auto &op : ops) {
        for (auto &in : op->inputs) {
            if (in->kind == ValueKind::PARAM) continue;
            Vertex::Connect(in->Vertex(), op);
        }
    }
    for (auto &out : outputs) Vertex::Connect(out->value->def.lock(), out);
}

struct StackRecord {
    VertexRef vertex;
    bool visited;
};

void Graph::Traverse(const std::function<void(const VertexRef &)> &func) const {
    // Initialize stack
    std::vector<StackRecord> stack;
    std::unordered_set<VertexRef> traversed;
    for (auto iter = outputs.rbegin(); iter != outputs.rend(); iter++)
        stack.push_back({*iter, false});

    // Iterate until no elements on stack
    while (!stack.empty()) {
        // Pop one vertex
        auto [vertex, visited] = stack.back();
        stack.pop_back();

        // Skip if this vertex is traversed before
        if (traversed.find(vertex) != traversed.end()) continue;

        // Apply function to vertex if it has been visited
        if (visited) {
            func(vertex);
            traversed.insert(vertex);
            continue;
        }

        // Otherwise add predecessors to stack
        stack.push_back({vertex, true});
        auto &preds = vertex->preds;
        for (auto iter = preds.rbegin(); iter != preds.rend(); iter++)
            stack.push_back({*iter, false});
    }
}

VertexRef VertexCloner::VisitInput(const InputRef &input) {
    auto newVal = VisitValue(input->value);
    auto newInput = std::make_shared<Input>(newVal);
    newVal->input = newInput;
    return newInput;
}

VertexRef VertexCloner::VisitOutput(const OutputRef &output) {
    auto &val = output->value;
    auto newVal = VisitValue(val);
    Visit(val->Vertex());
    return std::make_shared<Output>(newVal);
}

VertexRef VertexCloner::VisitOp(const OpRef &op) {
    auto newOp = std::make_shared<Op>(*op);
    for (auto &in : op->inputs) {
        auto newIn = VisitValue(in);
        newOp->inputs.push_back(newIn);
        newIn->uses.push_back(newOp);
        if (in->kind != ValueKind::PARAM) Visit(in->Vertex());
    }
    for (auto &out : op->outputs) {
        auto newOut = VisitValue(out);
        newOp->outputs.push_back(newOut);
        newOut->def = newOp;
    }
    return newOp;
}

ValueRef VertexCloner::VisitValue(const ValueRef &value) {
    if (Contains(valueMap, value)) return valueMap[value];
    auto newVal = std::make_shared<Value>(*value);
    valueMap.insert({value, newVal});
    return newVal;
}

class GraphCloner : public VertexCloner {
public:
    GraphCloner(const Graph &src, Graph &dst) : src(src), dst(dst) {}

    void Clone() {
        dst.name = src.name;
        for (auto &out : src.outputs) Visit(out);
        dst.ConnectVerts();
    }

    VertexRef VisitInput(const InputRef &input) override {
        auto newInput = VertexCloner::VisitInput(input);
        dst.inputs.push_back(As<Input>(newInput));
        return newInput;
    }

    VertexRef VisitOutput(const OutputRef &output) override {
        auto newOutput = VertexCloner::VisitOutput(output);
        dst.outputs.push_back(As<Output>(newOutput));
        return newOutput;
    }

    VertexRef VisitOp(const OpRef &op) override {
        auto newOp = VertexCloner::VisitOp(op);
        dst.ops.push_back(As<Op>(newOp));
        return op;
    }

    ValueRef VisitValue(const ValueRef &value) override {
        if (Contains(valueMap, value)) return valueMap[value];
        auto newVal = VertexCloner::VisitValue(value);
        if (newVal->kind == ValueKind::PARAM) dst.params.push_back(newVal);
        return newVal;
    }

protected:
    const Graph &src;
    Graph &dst;
};

Graph Graph::Clone() const {
    Graph dst;
    GraphCloner(*this, dst).Clone();
    return dst;
}

void Graph::Visualize(const std::string &dir, const std::string &format) const {
    // Define DOT graph
    DotCreator<VertexRef> creator(name);

    // Add vertices
    for (auto &in : inputs) creator.AddNode(in, in->value->name);
    for (auto &op : ops) creator.AddNode(op, op->type);
    for (auto &out : outputs) creator.AddNode(out, out->value->name);

    // Add edges
    for (auto &op : ops)
        for (auto &pred : op->preds) creator.AddEdge(pred, op);
    for (auto &out : outputs) creator.AddEdge(out->preds[0], out);

    // Compile
    creator.Render(dir, format);
}

}  // namespace hos