#pragma once

#include <onnx/onnx_pb.h>

#include <hmp/util.hpp>

namespace hmp {

using StrVec = std::vector<std::string>;

using I64Repeated = google::protobuf::RepeatedField<int64_t>;

inline std::string FmtInt(int i) { return fmt::format("{}", i); }

inline std::string FmtFloat(float f) { return fmt::format("{.2f}", f); }

inline std::string FmtStr(const std::string &s) {
    return fmt::format("'{}'", s);
}

inline std::string FmtDims(const I64Repeated &dims) {
    return JoinWithComma(Transform<StrVec>(dims, FmtInt), "(", ")");
}

std::string FmtAttrValue(const onnx::AttributeProto &attr);

}  // namespace hmp