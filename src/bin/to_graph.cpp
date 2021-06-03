#include <fstream>
#include <hos/core/hier.hpp>

using namespace hos;

int main(int argc, char const *argv[]) {
    // Initialize glog
    FLAGS_logtostderr = true;
    google::InitGoogleLogging(argv[0]);

    // Build computation graph from ONNX model
    std::ifstream ifs("../../model/nasnet_mobile.onnx", std::ifstream::binary);
    onnx::ModelProto model;
    model.ParseFromIstream(&ifs);
    Graph graph(model, "nasnet_mobile");

    // Build hierarchical graph
    HierGraph hier(graph);
    hier.VisualizeAll("../../out", "nasnet_hier");

    return 0;
}
