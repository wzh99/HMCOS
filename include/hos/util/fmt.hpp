#pragma once

#include <fmt/core.h>
#include <onnx/onnx_pb.h>

#include <hos/util/util.hpp>

namespace hos {

using I64Repeated = google::protobuf::RepeatedField<int64_t>;

inline std::string FmtInt(int64_t i) { return fmt::format("{}", i); }

inline std::string FmtFloat(float f) { return fmt::format("{:.2e}", f); }

std::string FmtStr(const std::string &s, char quote = '\'');

template <class Iterable, class F>
inline std::string FmtList(const Iterable &list, F fmt,
                           const char *prefix = "[", const char *suffix = "]",
                           const char *sep = ", ") {
    return Join(Transform<std::vector<std::string>>(list, fmt), sep, prefix,
                suffix);
}

inline std::string FmtTensorDims(const I64Repeated &dims) {
    if (dims.size() == 1)
        return fmt::format("({},)", dims[0]);
    else
        return FmtList(dims, FmtInt, "(", ")");
}

std::string FmtDataType(int32_t dtype);

inline std::string FmtTensorBrief(const onnx::TensorProto &tensor) {
    return fmt::format("Tensor<{}, {}>", FmtTensorDims(tensor.dims()),
                       FmtDataType(tensor.data_type()));
}

inline std::string FmtShapeDim(const onnx::TensorShapeProto_Dimension &dim) {
    return dim.has_dim_value() ? FmtInt(dim.dim_value()) : dim.dim_param();
}

inline std::string FmtShape(const onnx::TensorShapeProto &shape) {
    if (shape.dim().size() == 1)
        return fmt::format("({},)", FmtShapeDim(shape.dim(0)));
    else
        return FmtList(shape.dim(), FmtShapeDim, "(", ")");
}

inline std::string FmtTensorType(const onnx::TypeProto_Tensor &type) {
    return fmt::format("Tensor<{}, {}>", FmtShape(type.shape()),
                       FmtDataType(type.elem_type()));
}

std::string FmtAttrValue(const onnx::AttributeProto &attr);

}  // namespace hos