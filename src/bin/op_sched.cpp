#include <fmt/core.h>
#include <glog/logging.h>

#include <fstream>
#include <hos/sched/plan.hpp>
#include <hos/util/op.hpp>

using namespace hos;

int main(int argc, char const *argv[]) {
    // Initialize glog
    FLAGS_logtostderr = true;
    google::InitGoogleLogging(argv[0]);

    // Initialize op trait
    OpTraitRegistry::Init();

    // Read ONNX model
    std::ifstream ifs("../../model/mobilenet_v2.onnx", std::ifstream::binary);
    auto model = std::make_unique<onnx::ModelProto>();
    model->ParseFromIstream(&ifs);

    // Build graph and create schedule
    Graph graph(std::move(model), "mobilenet_v2");
    auto sched = ReversePostOrder(graph);
    auto ltVec = ComputeLifetime(sched, graph);
    // for (auto &lt : ltVec) lt.Print();

    Container container(0, 20);
    container.Place(1, 3, 3);
    container.Place(6, 3, 3);
    container.Lift(5);
    container.Print();
    
    return 0;
}
