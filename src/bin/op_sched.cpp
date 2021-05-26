#include <fmt/core.h>
#include <glog/logging.h>

#include <fstream>
#include <hos/sched/plan.hpp>
#include <hos/util/op.hpp>

using namespace hos;

int main(int argc, char const *argv[]) {
    // Initialize glog
    google::LogToStderr();
    google::InitGoogleLogging(argv[0]);

    // Initialize op trait
    OpTraitRegistry::Init();

    // Read ONNX model
    std::ifstream ifs("../../model/inception_v3.onnx", std::ifstream::binary);
    onnx::ModelProto model;
    model.ParseFromIstream(&ifs);
    ifs.close();

    // Build graph and create schedule
    Graph graph(model, "inception_v3");
    auto sched = ReversePostOrder(graph);
    auto stat = ComputeLifetime(sched, graph);
    auto plan = BestFit(stat);
    plan.Visualize("../../out", "inception_v3-rpo-best_fit");
    plan.Print();
    
    return 0;
}
