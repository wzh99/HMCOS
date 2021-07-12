#include <tensorflow/lite/simple_memory_arena.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <hos/sched/life.hpp>
#include <hos/sched/pass.hpp>
#include <hos/sched/plan.hpp>
#include <hos/sched/sched.hpp>

using namespace hos;
using namespace std::chrono;

static uint64_t computeArenaSize(const LifetimeStat &stat) {
    std::vector<tflite::ArenaAllocWithUsageInterval> allocs(stat.values.size());
    TfLiteContext ctx;
    tflite::SimpleMemoryArena arena(64);
    for (auto i = 0u; i < stat.values.size(); i++) {
        auto &val = stat.values[i];
        arena.Allocate(&ctx, 64, val.value->type.Size(), i, val.gen,
                       val.kill - 1, &allocs[i]);
    }
    return arena.RequiredBufferSize();
}

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
    LOG(INFO) << duration_cast<milliseconds>(clock.now() - begin).count();
    auto sched = HierarchicalSchedule(hier);
    LOG(INFO) << duration_cast<milliseconds>(clock.now() - begin).count();
    LOG(INFO) << EstimatePeak(sched, graph.inputs) / 1024;
    LOG(INFO) << computeArenaSize(ComputeLifetime(sched, graph)) / 1024;
    sched = ReversePostOrder(graph);
    LOG(INFO) << EstimatePeak(sched, graph.inputs) / 1024;
    LOG(INFO) << computeArenaSize(ComputeLifetime(sched, graph)) / 1024;

    return 0;
}
