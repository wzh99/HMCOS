#include <fstream>
#include <hos/core/graph.hpp>
#include <hos/core/value.hpp>

using namespace hos;

int main(int argc, char const *argv[]) {
    // Initialize glog
    FLAGS_logtostderr = true;
    google::InitGoogleLogging(argv[0]);

    // Build computation graph from ONNX model
    std::ifstream ifs("../model/nasnet_mobile.onnx", std::ifstream::binary);
    onnx::ModelProto model;
    model.ParseFromIstream(&ifs);
    Graph graph(model, "nasnet_mobile");
    auto newGraph = graph.Subgraph(
        [](const OpRef &op) {
            return op->name == "NASNet/cell_stem_0/cell_output/concat" ||
                   op->name == "NASNet/cell_stem_1/cell_output/concat";
        },
        "nasnet_block");

    // Visualize graph"
    newGraph.Visualize("../out");

    return 0;
}
