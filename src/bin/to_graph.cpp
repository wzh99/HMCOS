#include <fstream>
#include <hos/core/graph.hpp>
#include <hos/core/value.hpp>

int main(int argc, char const *argv[]) {
    FLAGS_logtostderr = true;
    FLAGS_minloglevel = google::GLOG_INFO;
    google::InitGoogleLogging(argv[0]);
    std::ifstream ifs("../../model/mobilenet_v2.onnx", std::ifstream::binary);
    auto model = std::make_unique<onnx::ModelProto>();
    model->ParseFromIstream(&ifs);
    hos::Graph graph(std::move(model));
    return 0;
}
