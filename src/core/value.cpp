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
    2 * sizeof(float),   // complex64
    2 * sizeof(double),  // complex128
    2,                   // bfloat16
};

#define GET_DATA_FUNC(field)                                      \
    [](const onnx::TensorProto &t) {                              \
        return std::make_pair(                                    \
            reinterpret_cast<const uint8_t *>(t.field().begin()), \
            reinterpret_cast<const uint8_t *>(t.field().end()));  \
    }

static std::pair<const uint8_t *, const uint8_t *> (*getDataFuncs[17])(
    const onnx::TensorProto &) = {
    nullptr,                     // undefined
    GET_DATA_FUNC(float_data),   // float32
    GET_DATA_FUNC(int32_data),   // uint8
    GET_DATA_FUNC(int32_data),   // int8
    GET_DATA_FUNC(int32_data),   // uint16
    GET_DATA_FUNC(int32_data),   // int16
    GET_DATA_FUNC(int32_data),   // int32
    GET_DATA_FUNC(int64_data),   // int64
    nullptr,                     // string
    GET_DATA_FUNC(int32_data),   // bool
    GET_DATA_FUNC(int32_data),   // float16
    GET_DATA_FUNC(double_data),  // float64
    GET_DATA_FUNC(uint64_data),  // uint32
    GET_DATA_FUNC(uint64_data),  // uint64
    GET_DATA_FUNC(float_data),   // complex64
    GET_DATA_FUNC(double_data),  // complex128
    GET_DATA_FUNC(int32_data)    // bfloat16
};

static std::vector<uint8_t> getTensorData(const onnx::TensorProto &tensor) {
    auto func = getDataFuncs[tensor.data_type()];
    if (!func) {
        LOG(FATAL) << fmt::format("Cannot get tensor data of type {}",
                                  FmtDataType(tensor.data_type()));
    }
    auto [begin, end] = func(tensor);
    return std::vector<uint8_t>(begin, end);
}

uint64_t TensorType::Count() const {
    return uint64_t(Accumulate(shape, std::multiplies<int64_t>(), 1ll));
}

uint64_t TensorType::Size() const { return Count() * scalarSize[dtype]; }

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
    value.data = getTensorData(tensor);
    return value;
}

VertexRef Value::Vertex() const {
    if (kind == ValueKind::INPUT)
        return input.lock();
    else if (kind == ValueKind::RESULT)
        return def.lock();
    else
        LOG(FATAL) << "Parameter value does not have corresponding vertex.";
}

}  // namespace hos