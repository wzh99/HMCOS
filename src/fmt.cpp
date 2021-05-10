#include <fmt/core.h>
#include <glog/logging.h>

#include <functional>
#include <hos/fmt.hpp>
#include <hos/util.hpp>
#include <unordered_map>

namespace hos {

using namespace onnx;

// Names must match those defined in `TensorProto::DataType`
static std::vector<std::string> dtypeNames{
    "undefined", "float32", "uint8",     "int8",       "uint16",   "int16",
    "int32",     "int64",   "string",    "bool",       "float16",  "float64",
    "uint32",    "uint64",  "complex64", "complex128", "bfloat16",
};

std::string FmtDataType(int32_t dtype) { return dtypeNames[dtype]; }

static std::unordered_map<AttributeProto::AttributeType,
                          std::string (*)(const AttributeProto &)>
    attrFmtFuncMap{
        {AttributeProto::INT, [](auto a) { return FmtInt(a.i()); }},
        {AttributeProto::FLOAT, [](auto a) { return FmtFloat(a.f()); }},
        {AttributeProto::STRING, [](auto a) { return FmtStr(a.s()); }},
        {AttributeProto::TENSOR, [](auto a) { return FmtTensorBrief(a.t()); }},
        {AttributeProto::INTS,
         [](auto a) { return FmtList(a.ints(), FmtInt); }},
        {AttributeProto::FLOATS,
         [](auto a) { return FmtList(a.floats(), FmtFloat); }},
        {AttributeProto::STRINGS,
         [](auto a) { return FmtList(a.strings(), FmtStr); }},
        {AttributeProto::TENSORS,
         [](auto a) { return FmtList(a.tensors(), FmtTensorBrief); }},
    };

std::string FmtAttrValue(const AttributeProto &attr) {
    auto type = attr.type();
    if (!Contains(attrFmtFuncMap, type)) {
        LOG(ERROR) << fmt::format("Cannot format attribute type {}.", type);
        return "";
    } else
        return attrFmtFuncMap[type](attr);
}

}  // namespace hos