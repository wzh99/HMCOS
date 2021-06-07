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
    std::ifstream ifs("../../model/nasnet_mobile.onnx", std::ifstream::binary);
    onnx::ModelProto model;
    model.ParseFromIstream(&ifs);
    Graph graph(model, "nasnet_mobile");

    // Build hierarchical graph
    HierGraph hier(graph);
    RunPass<JoinSequencePass, MakeGroupPass>(hier);
    // hier.VisualizeAll("../out", "nasnet_hier");
    hier.VisualizeTop("../../out", "nasnet-top");
    hier.VisualizeDom(false, "../../out", "nasnet-dom");

    return 0;
}
