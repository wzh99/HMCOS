#pragma once

#include <onnx/onnx_pb.h>

#include <hos/core/rtti.hpp>

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
struct Input;
struct Vertex;
using OpRef = std::shared_ptr<Op>;
using InputRef = std::shared_ptr<Input>;
using VertexRef = std::shared_ptr<Vertex>;

struct Value {
    /// Common fields of all kinds of values
    ValueKind kind;
    std::string name;
    TensorType type;

    /// Kind-specific fields.

    /// Valid for input. Stores shared pointer to corresponding input vertex.
    InputRef input;
    /// Valid for parameter. Stores pointer to tensor data.
    const onnx::TensorProto *data = nullptr;
    /// Valid for result. Stores shared pointer to op which defines this value.
    OpRef def = nullptr;
    /// Valid for result. Stores shared pointers to ops that use (take as input)
    /// this value.
    std::vector<OpRef> uses;

    static Value CreateInput(const onnx::ValueInfoProto &info);
    static Value CreateParam(const onnx::TensorProto &tensor);
    static Value CreateResult(const onnx::ValueInfoProto &info);

    /// Return the vertex in graph where this value is defined.
    VertexRef GetVertex() const;
};

using ValueRef = std::shared_ptr<Value>;

}  // namespace hos
