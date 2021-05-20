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
    std::ifstream ifs("../../model/mobilebert.onnx", std::ifstream::binary);
    auto model = std::make_unique<onnx::ModelProto>();
    model->ParseFromIstream(&ifs);

    // Build graph and create schedule
    Graph graph(std::move(model), "mobilebert");
    auto sched = ReversePostOrder(graph);
    auto stat = ComputeLifetime(sched, graph);
    auto plan = BestFit(stat);
    plan.Visualize("../../out", "mobilebert-rpo-best_fit");
    plan.Print();
    
    return 0;
}
