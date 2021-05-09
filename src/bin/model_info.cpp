#include <fmt/core.h>
#include <glog/logging.h>
#include <onnx/onnx_pb.h>

#include <fstream>
#include <hmp/fmt.hpp>
#include <stdexcept>

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
    std::ifstream ifs("../model/resnet_v1.onnx", std::ifstream::binary);
    if (!ifs) LOG(FATAL) << "Cannot open model file.";
    onnx::ModelProto model;
    model.ParseFromIstream(&ifs);
    if (!model.has_graph()) LOG(FATAL) << ("Cannot read graph from model.");
    auto &graph = model.graph();

    // Print nodes
    auto prefix =
        "sequential/keras_layer/StatefulPartitionedCall/"
        "StatefulPartitionedCall/StatefulPartitionedCall/predict/";
    fmt::print("Nodes:\n");
    for (auto &node : graph.node()) {
        fmt::print("{} ", node.op_type());
        auto inputNames = hmp::Transform<hmp::StrVec>(
            node.input(),
            [&](auto &input) { return shortValueName(input, prefix); });
        fmt::print("{} ", hmp::JoinWithComma(inputNames, "(", ")"));
        auto attrItems = hmp::Transform<hmp::StrVec>(
            node.attribute(), [](auto &attr) {
                return fmt::format("{}={}", attr.name(),
                                   hmp::FmtAttrValue(attr));
            });
        fmt::print("{} ", hmp::JoinWithComma(attrItems, "{", "}"));
        auto outputNames = hmp::Transform<hmp::StrVec>(
            node.output(), 
            [&](auto &output) { return shortValueName(output, prefix); }
        );
        fmt::print("-> {}", hmp::JoinWithComma(outputNames, "(", ")"));
        fmt::print("\n");
    }
}