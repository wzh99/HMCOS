#include <fstream>
#include <hos/core/graph.hpp>
#include <hos/core/value.hpp>

using namespace hos;

int main(int argc, char const *argv[]) {
    // Initialize glog
    FLAGS_logtostderr = true;
    google::InitGoogleLogging(argv[0]);

    // Build computation graph from ONNX model
    std::ifstream ifs("../model/mobilenet_v2.onnx", std::ifstream::binary);
    auto model = std::make_unique<onnx::ModelProto>();
    model->ParseFromIstream(&ifs);
    Graph graph(std::move(model), "mobilenet_v2");

    // Visualize graph
    graph.Traverse([](VertexRef v) {
        // fmt::print("{}\n", v->GetTypeIndex());
        if (Is<Op>(v)) {
            auto op = As<Op>(v);
            fmt::print("{}\n", op->GetType());
        }
    });
    // graph.Visualize("../out");

    return 0;
}
