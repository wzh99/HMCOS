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
    std::ifstream ifs("../../model/nasnet_mobile.onnx", std::ifstream::binary);
    onnx::ModelProto model;
    model.ParseFromIstream(&ifs);
    ifs.close();

    // Build graph and create schedule
    Graph graph(model, "nasnet_mobile");
    graph = graph.Subgraph(
        [](const OpRef &op) {
            return op->name == "NASNet/cell_stem_0/cell_output/concat" ||
                   op->name == "NASNet/cell_stem_1/concat";
        },
        "nasnet_block");
    BruteForceSearch(
        graph,
        [&](const OpSeq &seq) {
            return double(ComputeLifetime(seq, graph).Peak());
        },
        [&](const OpSeq &seq, double metric) { 
            fmt::print("{}\n", metric); 
        });

    return 0;
}
