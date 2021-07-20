#include <tensorflow/lite/simple_memory_arena.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <hos/sched/life.hpp>
#include <hos/sched/pass.hpp>
#include <hos/sched/plan.hpp>
#include <hos/sched/sched.hpp>
#include <hos/util/viz.hpp>

using namespace hos;
using namespace std::chrono;

#define TIME_CODE(code)                                                        \
    {                                                                          \
        auto _begin = system_clock::now();                                     \
        code;                                                                  \
        auto _dur =                                                            \
            duration_cast<milliseconds>(system_clock::now() - _begin).count(); \
        LOG(INFO) << fmt::format("{} ms", _dur);                               \
    }

static void sampleSchedDistrib(const Graph &graph, size_t nSamples,
                               const std::string &dir) {
    std::mt19937 rng;
    HistoPlot plot(graph.name + "-distrib");
    for (auto i = 0u; i < nSamples; i++) {
        auto sched = RandomSample(graph, rng);
        auto peak = EstimatePeak(sched, graph.inputs);
        plot.Append(float(peak / 1024));
    }
    plot.Render(dir, "pdf");
}

static void sampleLowestPeakSched(const Graph &graph, size_t nSamples,
                                  const std::string &dir) {
    std::mt19937 rng;
    uint64_t minPeak = UINT64_MAX;
    std::vector<OpRef> minSched;
    for (auto i = 0u; i < nSamples; i++) {
        auto sched = RandomSample(graph, rng);
        auto peak = EstimatePeak(sched, graph.inputs);
        if (peak < minPeak) {
            minPeak = peak;
            minSched = sched;
        }
    }
    ComputeLifetime(minSched, graph).Plot(dir, graph.name + "-usage");
    PlotSchedule(minSched, graph, dir, graph.name + "-min");
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
    LOG(INFO) << EstimatePeak(sched, graph.inputs) / 1024;
    LOG(INFO) << computeArenaSize(ComputeLifetime(sched, graph)) / 1024;
    sched = ReversePostOrder(graph);
    LOG(INFO) << EstimatePeak(sched, graph.inputs) / 1024;
    LOG(INFO) << computeArenaSize(ComputeLifetime(sched, graph)) / 1024;
    // sampleLowestPeakSched(graph, 10000, argv[2]);

    return 0;
}
