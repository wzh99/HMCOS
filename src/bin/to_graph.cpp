#include <filesystem>
#include <fstream>
#include <hos/sched/pass.hpp>
#include <hos/util/op.hpp>

using namespace hos;

int main(int argc, char const *argv[]) {
    // Initialize glog
    google::LogToStderr();
    FLAGS_minloglevel = 0;
    google::InitGoogleLogging(argv[0]);

    // Build computation graph from ONNX model
    std::ifstream ifs(argv[1], std::ifstream::binary);
    onnx::ModelProto model;
    model.ParseFromIstream(&ifs);
    ifs.close();
    auto name = std::filesystem::path(argv[1]).stem().string();
    Graph graph(model, name);
    graph.Visualize(argv[2]);
    model.Clear();

    // Build hierarchical graph
    HierGraph hier(graph);
    RunPass<JoinSequencePass, MakeGroupPass>(hier);
    hier.VisualizeAll(argv[2], name + "-all");
    hier.VisualizeTop(argv[2], name + "-top");

    return 0;
}
