#include <fmt/core.h>
#include <glog/logging.h>
#include <onnx/onnx_pb.h>
#include <onnx/shape_inference/implementation.h>

#include <fstream>
#include <hmcos/util/fmt.hpp>
#include <stdexcept>

using namespace hmcos;

int main(int argc, char *argv[]) {
    // Init logging
    FLAGS_logtostderr = true;
    google::InitGoogleLogging(argv[0]);

    // Read graph from file
    std::string path = "../../model/mobilenet_v2.onnx";
    std::ifstream ifs(path, std::ifstream::binary);
    if (!ifs) LOG(FATAL) << "Cannot open model file.";
    onnx::ModelProto model;
    model.ParseFromIstream(&ifs);
    if (!model.has_graph()) LOG(FATAL) << ("Cannot read graph from model.");
    auto &graph = model.graph();

    // Print inputs and outputs
    auto fmtValue = [&](const onnx::ValueInfoProto &info) {
        return fmt::format("{}: {}", info.name(),
                           FmtTensorType(info.type().tensor_type()));
    };
    fmt::print("\nInputs: \n");
    for (auto &input : graph.input()) fmt::print("{}\n", fmtValue(input));
    fmt::print("\nOutputs: \n");
    for (auto &output : graph.output()) fmt::print("{}\n", fmtValue(output));

    // Print nodes
    fmt::print("\nNodes:\n");
    for (auto &node : graph.node()) {
        fmt::print("{} ", node.op_type());
        fmt::print("{} ", JoinWithComma(node.input(), "(", ")"));
        fmt::print("{} ", FmtList(
                              node.attribute(),
                              [](auto &attr) {
                                  return fmt::format("{}={}", attr.name(),
                                                     FmtAttrValue(attr));
                              },
                              "{", "}"));
        fmt::print("-> {}", JoinWithComma(node.output(), "(", ")"));
        fmt::print("\n");
    }

    // Print parameters
    fmt::print("\nParameters:\n");
    for (auto &tensor : graph.initializer())
        fmt::print("{}: {}\n", tensor.name(), FmtTensorBrief(tensor));

    // Print values
    fmt::print("\nValues:\n");
    for (auto &info : graph.value_info()) fmt::print("{}\n", fmtValue(info));

    return 0;
}