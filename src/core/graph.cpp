#include <hos/core/graph.hpp>
#include <unordered_map>

namespace hos {

Graph::Graph(std::unique_ptr<onnx::ModelProto> &&model)
    : model(std::move(model)) {
    // Build name-value map
    std::unordered_map<std::string, ValueRef> nameToVal;
    auto &graph = this->model->graph();
    // Inputs
    for (auto &info : graph.input()) {
        auto val = std::make_shared<Value>(Value::CreateInput(info));
        inputs.push_back(std::make_shared<Input>(val));
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
    
}

}  // namespace hos