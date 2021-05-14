#include <hos/core/graph.hpp>
#include <hos/core/value.hpp>
#include <hos/util/fmt.hpp>

namespace hos {

TensorType TensorType::FromTensor(const onnx::TensorProto &tensor) {
    auto &dims = tensor.dims();
    return TensorType{std::vector<int64_t>(dims.begin(), dims.end()),
                      DataType(tensor.data_type())};
}

TensorType TensorType::FromType(const onnx::TypeProto_Tensor &type) {
    std::vector<int64_t> shape;
    for (auto &dim : type.shape().dim()) {
        if (!dim.has_dim_value())
            LOG(FATAL) << fmt::format("{} is not a dimension value.",
                                      dim.dim_param());
        shape.push_back(dim.dim_value());
    }
    return TensorType{shape, DataType(type.elem_type())};
}

Value Value::CreateInput(const onnx::ValueInfoProto &info) {
    Value value;
    value.kind = ValueKind::INPUT;
    value.name = info.name();
    value.type = TensorType::FromType(info.type().tensor_type());
    return value;
}

Value Value::CreateResult(const onnx::ValueInfoProto &info) {
    Value value;
    value.kind = ValueKind::RESULT;
    value.name = info.name();
    value.type = TensorType::FromType(info.type().tensor_type());
    return value;
}

Value Value::CreateParam(const onnx::TensorProto &tensor) {
    Value value;
    value.kind = ValueKind::PARAM;
    value.name = tensor.name();
    value.type = TensorType::FromTensor(tensor);
    value.data = &tensor;
    return value;
}

VertexRef Value::GetVertex() const {
    if (kind == ValueKind::INPUT)
        return input;
    else if (kind == ValueKind::RESULT)
        return def;
    else
        LOG(FATAL) << "Parameter value does not have corresponding vertex.";
}

}  // namespace hos