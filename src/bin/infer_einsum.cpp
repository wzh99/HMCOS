#include <glog/logging.h>
#include <onnx/defs/schema.h>
#include <onnx/onnx_pb.h>
#include <onnx/shape_inference/implementation.h>

#include <fstream>
#include <functional>
#include <hos/util/fmt.hpp>

#ifdef _FUNCTIONAL_
#include <onnx/defs/shape_inference.h>
#endif

void inferEinsum(onnx::InferenceContext &ctx) {
    // Parse equation
    auto &eq = ctx.getAttribute("equation")->s();
    auto arrowIdx = eq.find("->");
    auto lhs = eq.substr(0, arrowIdx), rhs = eq.substr(arrowIdx + 2);

    // Build index-dim map
    std::unordered_map<char, int64_t> idxToDim;
    for (auto i = 0u; i < ctx.getNumInputs(); i++) {
        auto &inShape = ctx.getInputType(i)->tensor_type().shape();
        std::string idxStr;
        if (i == ctx.getNumInputs() - 1)
            idxStr = lhs;
        else {
            auto commaIdx = eq.find(',');
            idxStr = lhs.substr(0, commaIdx);
            lhs.erase(0, idxStr.size() + 1);
        }
        LOG_ASSERT(idxStr.size() == inShape.dim_size());
        for (auto d = 0; d < idxStr.size(); d++) {
            auto c = idxStr[d];
            idxToDim.insert({c, inShape.dim(d).dim_value()});
        }
    }

    // Infer output type
    auto elemTy = ctx.getInputType(0)->tensor_type().elem_type();
    auto outTy = ctx.getOutputType(0)->mutable_tensor_type();
    outTy->set_elem_type(elemTy);
    auto outShape = outTy->mutable_shape();
    for (auto c : rhs) outShape->add_dim()->set_dim_value(idxToDim[c]);
    fmt::print("Type inferred for Einsum op: {}\n", hos::FmtTensorType(*outTy));
}

int main(int argc, char const *argv[]) {
    // Init logging
    google::LogToStderr();
    google::InitGoogleLogging(argv[0]);

    // Read graph from file
    std::string path = "../../model/mobilebert.onnx";
    std::ifstream ifs(path, std::ifstream::binary);
    if (!ifs) LOG(FATAL) << "Cannot open model file.";
    onnx::ModelProto model;
    model.ParseFromIstream(&ifs);
    ifs.close();
    if (!model.has_graph()) LOG(FATAL) << "Cannot read graph from model.";
    auto graph = model.mutable_graph();

    // Override einsum inference function
    auto schema =
        const_cast<onnx::OpSchema *>(onnx::OpSchemaRegistry::Schema("Einsum"));
    schema->TypeAndShapeInferenceFunction(inferEinsum);
    onnx::shape_inference::InferShapes(model, true);

    // Save model
    std::ofstream ofs(path, std::ofstream::binary);
    model.SerializeToOstream(&ofs);

    return 0;
}
