#include <fstream>
#include <hos/sched/pass.hpp>
#include <hos/util/op.hpp>

using namespace hos;

int main(int argc, char const *argv[]) {
    // Initialize glog
    google::LogToStderr();
    FLAGS_minloglevel = 0;
    google::InitGoogleLogging(argv[0]);

    // Initialize op traits
    OpTraitRegistry::Init();

    // Build computation graph from ONNX model
    std::ifstream ifs(argv[1], std::ifstream::binary);
    onnx::ModelProto model;
    model.ParseFromIstream(&ifs);
    ifs.close();
    Graph graph(model, "nasnet_mobile");
    model.Clear();

    // Build hierarchical graph
    HierGraph hier(graph);
    RunPass<JoinSequencePass, MakeGroupPass>(hier);
    hier.VisualizeAll(argv[2], "nasnet-all");
    hier.VisualizeTop(argv[2], "nasnet-top");

    return 0;
}
