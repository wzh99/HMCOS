#pragma once

#include <onnx/onnx_pb.h>

#include <hos/core/obj.hpp>

namespace hos {

/// Consistent with `TensorProto_DataType` in <onnx/onnx_pb.h>
enum DataType {
    UNDEFINED = 0,
    FLOAT = 1,
    UINT8 = 2,
    INT8 = 3,
    UINT16 = 4,
    INT16 = 5,
    INT32 = 6,
    INT64 = 7,
    STRING = 8,
    BOOL = 9,
    FLOAT16 = 10,
    DOUBLE = 11,
    UINT32 = 12,
    UINT64 = 13,
    COMPLEX64 = 14,
    COMPLEX128 = 15,
    BFLOAT16 = 16,
};

/// Internal storage of tensor type. In this project, all tensors must
/// have concrete shapes.
struct TensorType {
    std::vector<int64_t> shape;
    DataType dtype;

    static TensorType FromTensor(const onnx::TensorProto &tensor);
    static TensorType FromType(const onnx::TypeProto_Tensor &type);
};

enum class ValueKind {
    /// Input values of the model
    INPUT,
    /// Parameters
    PARAM,
    /// Intermediate or final results
    RESULT,
};

struct Op;
using OpRef = std::shared_ptr<Op>;

struct Value {
    /// Common fields of all kinds of values
    ValueKind kind;
    std::string name;
    TensorType type;

    /// Kind-specific fields
    /// Valid for parameter. Stores pointer to tensor data.
    const onnx::TensorProto *data = nullptr;
    /// Valid for intermediate. Stores counted reference to operator which
    /// defines this value.
    OpRef def;
    std::vector<OpRef> uses;

    static Value CreateInput(const onnx::ValueInfoProto &info);
    static Value CreateParam(const onnx::TensorProto &tensor);
    static Value CreateResult(const onnx::ValueInfoProto &info);
};

using ValueRef = std::shared_ptr<Value>;

}  // namespace hos
