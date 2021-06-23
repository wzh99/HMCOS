#include <filesystem>
#include <fstream>
#include <chrono>
#include <hos/sched/life.hpp>
#include <hos/sched/pass.hpp>
#include <hos/sched/sched.hpp>

using namespace hos;
using namespace std::chrono;

int main(int argc, char const *argv[]) {
    // Initialize glog
    FLAGS_minloglevel = 0;
    google::LogToStderr();
    google::InitGoogleLogging(argv[0]);

    // Build compitation graph from ONNX model
    std::ifstream ifs(argv[1], std::ifstream::binary);
    onnx::ModelProto model;
    model.ParseFromIstream(&ifs);
    ifs.close();
    Graph graph(model, std::filesystem::path(argv[1]).stem().string());
    model.Clear();

    // Build hierarchical graph
    HierGraph hier(graph);
    system_clock clock;
    auto begin = clock.now();
    RunPass<JoinSequencePass, MakeGroupPass>(hier);
    auto sched = HierarchicalSchedule(hier);
    LOG(INFO) << duration_cast<milliseconds>(clock.now() - begin).count();
    LOG(INFO) << EstimatePeak(sched, graph.inputs) / 1024;
    LOG(INFO) << EstimatePeak(ReversePostOrder(graph), graph.inputs) / 1024;

    return 0;
}
