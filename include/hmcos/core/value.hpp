#pragma once

#include <onnx/onnx_pb.h>

#include <hmcos/util/rtti.hpp>
#include <hmcos/util/util.hpp>

namespace hmcos {

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

    /// Number of elements in this tensor
    uint64_t Count() const;
    /// Size of this tensor in memory
    uint64_t Size() const;

    bool operator==(const TensorType &other) const;
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
using VertexRef = std::shared_ptr<Vertex>;

struct Value {
    /// Common fields of all kinds of values
    ValueKind kind;
    std::string name;
    TensorType type;

    /// Kind-specific fields.

    /// Valid for input. Stores shared pointer to corresponding input vertex.
    std::weak_ptr<Input> input;
    /// Valid for parameter. Stores pointer to tensor data.
    std::vector<uint8_t> data;
    /// Valid for result. Stores shared pointer to op which defines this value.
    std::weak_ptr<Op> def;
    /// Valid for result. Stores shared pointers to ops that use (take as input)
    /// this value. An op may appear multiple times if it uses this value more
    /// than once.
    std::vector<std::weak_ptr<Op>> uses;

    static Value CreateInput(const onnx::ValueInfoProto &info);
    static Value CreateParam(const onnx::TensorProto &tensor);
    static Value CreateResult(const onnx::ValueInfoProto &info);

    Value() = default;

    /// Clone from a value
    /// Usually used in vertex cloning, so all weak references to graph vertices
    /// are not copied.
    Value(const Value &other)
        : kind(other.kind),
          name(other.name),
          type(other.type),
          data(other.data) {}

    /// Return the vertex in graph where this value is defined.
    VertexRef Vertex() const;
};

using ValueRef = std::shared_ptr<Value>;

}  // namespace hmcos
