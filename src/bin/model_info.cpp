#include <fmt/core.h>
#include <glog/logging.h>
#include <onnx/onnx_pb.h>

#include <fstream>
#include <stdexcept>

int main(int argc, char *argv[]) {
    // Init logging
    FLAGS_logtostderr = true;
    google::InitGoogleLogging(argv[0]);

    // Read graph from file
    std::ifstream ifs("../../model/resnet_v1.onnx", std::ifstream::binary);
    if (!ifs) LOG(FATAL) << "Cannot open model file.";
    onnx::ModelProto model;
    model.ParseFromIstream(&ifs);
    if (!model.has_graph()) LOG(FATAL) << ("Cannot read graph from model.");
    auto &graph = model.graph();

    // Print nodes
    fmt::print("Nodes:\n");
    for (auto &node : graph.node()) {
        fmt::print("{} ", node.op_type());
        for (auto &input : node.input())
            fmt::print("{} ", input);
        fmt::print("\n");
    }
}