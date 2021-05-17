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

static uint64_t scalarSize[] = {
    0,
    sizeof(float),
    sizeof(uint8_t),
    sizeof(int8_t),
    sizeof(uint16_t),
    sizeof(int16_t),
    sizeof(int32_t),
    sizeof(int64_t),
    sizeof(std::string),
    sizeof(bool),
    2,  // float16
    sizeof(double),
    sizeof(uint32_t),
    sizeof(uint64_t),
    8,   // complex64
    16,  // complex128
    2,   // bfloat16
};

uint64_t TensorType::Size() const {
    auto nElem = Accumulate(shape, std::multiplies<int64_t>(), 1ll);
    return uint64_t(nElem) * scalarSize[dtype];
}

bool TensorType::operator==(const TensorType &other) const {
    if (this->dtype != other.dtype) return false;
    if (this->shape.size() != other.shape.size()) return false;
    for (auto i = 0u; i < shape.size(); i++)
        if (this->shape[i] != other.shape[i]) return false;
    return true;
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