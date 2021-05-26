#include <fstream>
#include <hos/core/graph.hpp>
#include <hos/core/value.hpp>

using namespace hos;

int main(int argc, char const *argv[]) {
    // Initialize glog
    FLAGS_logtostderr = true;
    google::InitGoogleLogging(argv[0]);

    // Build computation graph from ONNX model
    std::ifstream ifs("../../model/mobilebert.onnx", std::ifstream::binary);
    onnx::ModelProto model;
    model.ParseFromIstream(&ifs);
    Graph graph(model, "mobilebert");

    // Visualize graph
    graph.Visualize("../../out");

    return 0;
}
