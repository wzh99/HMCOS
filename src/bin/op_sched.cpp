#include <tensorflow/lite/simple_memory_arena.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <hmcos/sched/life.hpp>
#include <hmcos/sched/pass.hpp>
#include <hmcos/sched/plan.hpp>
#include <hmcos/sched/sched.hpp>
#include <hmcos/util/viz.hpp>

using namespace hmcos;
using namespace std::chrono;

#define TIME_CODE(code)                                                        \
    {                                                                          \
        auto _begin = system_clock::now();                                     \
        code;                                                                  \
        auto _dur =                                                            \
            duration_cast<milliseconds>(system_clock::now() - _begin).count(); \
        LOG(INFO) << fmt::format("{} ms", _dur);                               \
    }

static uint64_t computeArenaSize(const LifetimeStat &stat) {
    std::vector<tflite::ArenaAllocWithUsageInterval> allocs(stat.values.size());
    TfLiteContext ctx;
    tflite::SimpleMemoryArena arena(64);
    for (auto [i, val] : EnumRange(stat.values))
        arena.Allocate(&ctx, 64, val.value->type.Size(), i, val.gen,
                       val.kill - 1, &allocs[i]);
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

    // Schedule hierarchical graph
    std::vector<OpRef> sched;
    TIME_CODE(sched = HierarchicalSchedule(graph);)
    LOG(INFO) << "HMCOS Peak: " << EstimatePeak(sched, graph.inputs) / 1024 << " KB";
    LOG(INFO) << "HMCOS Arena Size: " << computeArenaSize(ComputeLifetime(sched, graph)) / 1024 << " KB";
    sched = ReversePostOrder(graph);
    LOG(INFO) << "RPO Peak: " << EstimatePeak(sched, graph.inputs) / 1024 << " KB";
    LOG(INFO) << "RPO Arena Size: " << computeArenaSize(ComputeLifetime(sched, graph)) / 1024 << " KB";

    return 0;
}
