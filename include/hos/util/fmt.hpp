#pragma once

#include <onnx/onnx_pb.h>

#include <hos/util/util.hpp>

namespace hos {

using StrVec = std::vector<std::string>;

using I64Repeated = google::protobuf::RepeatedField<int64_t>;

inline std::string FmtInt(int64_t i) { return fmt::format("{}", i); }

inline std::string FmtFloat(float f) { return fmt::format("{:.2e}", f); }

inline std::string FmtStr(const std::string &s) {
    return fmt::format("'{}'", s);
}

inline std::string FmtDims(const I64Repeated &dims) {
    if (dims.size() == 1)
        return fmt::format("({},)", dims[0]);
    else
        return JoinWithComma(Transform<StrVec>(dims, FmtInt), "(", ")");
}

std::string FmtDataType(int32_t dtype);

inline std::string FmtTensorBrief(const onnx::TensorProto &tensor) {
    return fmt::format("Tensor<{}, {}>", FmtDims(tensor.dims()),
                       FmtDataType(tensor.data_type()));
}

template <typename Iterable, typename F>
inline std::string FmtList(const Iterable &list, F fmt) {
    return JoinWithComma(Transform<StrVec>(list, fmt), "[", "]");
}

std::string FmtAttrValue(const onnx::AttributeProto &attr);

}  // namespace hos