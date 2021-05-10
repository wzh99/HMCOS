#include <fmt/core.h>
#include <glog/logging.h>
#include <onnx/common/ir.h>
#include <onnx/onnx_pb.h>

#include <fstream>
#include <hos/fmt.hpp>
#include <stdexcept>

using namespace hos;

std::string shortValueName(const std::string &valueName,
                           const std::string &prefix) {
    auto modelNamePos = valueName.find(prefix);
    return modelNamePos == std::string::npos
               ? valueName
               : valueName.substr(modelNamePos + prefix.length());
}

int main(int argc, char *argv[]) {
    // Init logging
    FLAGS_logtostderr = true;
    google::InitGoogleLogging(argv[0]);

    // Read graph from file
    std::ifstream ifs("../../model/mobilenet_v2.onnx", std::ifstream::binary);
    if (!ifs) LOG(FATAL) << "Cannot open model file.";
    onnx::ModelProto model;
    model.ParseFromIstream(&ifs);
    if (!model.has_graph()) LOG(FATAL) << ("Cannot read graph from model.");
    auto &graph = model.graph();

    // Print nodes
    std::string prefix =
        "sequential/keras_layer/StatefulPartitionedCall/"
        "StatefulPartitionedCall/StatefulPartitionedCall/predict/";
    fmt::print("Nodes:\n");
    for (auto &node : graph.node()) {
        fmt::print("{} ", node.op_type());
        auto inputNames = Transform<StrVec>(node.input(), [&](auto &input) {
            return shortValueName(input, prefix);
        });
        fmt::print("{} ", JoinWithComma(inputNames, "(", ")"));
        auto attrItems = Transform<StrVec>(node.attribute(), [](auto &attr) {
            return fmt::format("{}={}", attr.name(), FmtAttrValue(attr));
        });
        fmt::print("{} ", JoinWithComma(attrItems, "{", "}"));
        auto outputNames = Transform<StrVec>(node.output(), [&](auto &output) {
            return shortValueName(output, prefix);
        });
        fmt::print("-> {}", JoinWithComma(outputNames, "(", ")"));
        fmt::print("\n");
    }

    // Print parameters
    fmt::print("\nParameters:\n");
    for (auto &tensor : graph.initializer())
        fmt::print("{}: {}\n", shortValueName(tensor.name(), prefix),
                   FmtTensorBrief(tensor));
}