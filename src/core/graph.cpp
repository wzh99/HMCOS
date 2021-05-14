#include <hos/core/graph.hpp>
#include <hos/util/fmt.hpp>
#include <hos/util/viz.hpp>
#include <unordered_map>

namespace hos {

Graph::Graph(std::unique_ptr<onnx::ModelProto> &&model, const std::string &name)
    : model(std::move(model)) {
    // Create name of this graph
    auto &graph = this->model->graph();
    this->name = name.size() == 0 ? model->graph().name() : name;

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
    for (auto &op : ops) {
        for (auto &in : op->inputs) {
            if (in->kind == ValueKind::PARAM) continue;
            Vertex::Connect(in->GetVertex(), op);
        }
    }
    for (auto &out : outputs) Vertex::Connect(out->value->def, out);
}

}  // namespace hos