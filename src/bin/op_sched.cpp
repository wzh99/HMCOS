#include <hos/util/op.hpp>
#include <hos/sched/order.hpp>
#include <fstream>
#include <glog/logging.h>
#include <fmt/core.h>

using namespace hos;

int main(int argc, char const *argv[]) {
    FLAGS_logtostderr = true;
    google::InitGoogleLogging(argv[0]);
    OpTraitRegistry::Init();
    std::ifstream ifs("../model/mobilenet_v2.onnx", std::ifstream::binary);
    auto model = std::make_unique<onnx::ModelProto>();
    model->ParseFromIstream(&ifs);
    Graph graph(std::move(model), "mobilenet_v2");
    auto sched = ReversePostOrder(graph);
    sched.Print();
    return 0;
}
