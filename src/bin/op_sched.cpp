#include <fmt/core.h>
#include <glog/logging.h>

#include <fstream>
#include <hos/sched/plan.hpp>
#include <hos/util/op.hpp>

using namespace hos;

int main(int argc, char const *argv[]) {
    // Initialize glog
    FLAGS_minloglevel = 0;
    google::LogToStderr();
    google::InitGoogleLogging(argv[0]);

    // Initialize op trait
    OpTraitRegistry::Init();

    // Read ONNX model
    std::ifstream ifs("../model/nasnet_mobile.onnx", std::ifstream::binary);
    onnx::ModelProto model;
    model.ParseFromIstream(&ifs);
    ifs.close();

    // Build graph and create schedule
    Graph graph(model, "nasnet_mobile");
    graph = graph.Subgraph(
        [](const OpRef &op) {
            return op->name == "NASNet/cell_stem_1/cell_output/concat";
        },
        "nasnet_block");
    BruteForceSearch(
        graph,
        [&](const OpSeq &seq) { return EstimatePeak(seq, graph.inputs); },
        [&](const OpSeq &seq, uint64_t metric) {
            for (auto &op : seq) fmt::print("{}\n", op->name);
            fmt::print("Peak: {}\n", metric);
        });

    return 0;
}
