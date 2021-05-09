#include <fmt/core.h>
#include <glog/logging.h>

#include <functional>
#include <hmp/fmt.hpp>
#include <hmp/util.hpp>
#include <unordered_map>

namespace hmp {

using namespace onnx;

static std::unordered_map<AttributeProto::AttributeType,
                          std::string (*)(const AttributeProto &)>
    attrFmtFuncMap{
        {AttributeProto::INT, [](auto a) { return FmtInt(a.i()); }},
        {AttributeProto::FLOAT, [](auto a) { return FmtFloat(a.f()); }},
        {AttributeProto::STRING, [](auto a) { return FmtStr(a.s()); }},
        {AttributeProto::INTS,
         [](auto a) {
             return JoinWithComma(Transform<StrVec>(a.ints(), FmtInt), "[",
                                  "]");
         }},
        {AttributeProto::FLOATS,
         [](auto a) {
             return JoinWithComma(Transform<StrVec>(a.floats(), FmtFloat), "[",
                                  "]");
         }},
        {AttributeProto::STRINGS,
         [](auto a) {
             return JoinWithComma(Transform<StrVec>(a.strings(), FmtStr), "[",
                                  "]");
         }},
    };

std::string FmtAttrValue(const AttributeProto &attr) {
    auto type = attr.type();
    if (!Contains(attrFmtFuncMap, type)) {
        LOG(ERROR) << fmt::format("Cannot format attribute type {}.", type);
        return "";
    } else
        return attrFmtFuncMap[type](attr);
}

}  // namespace hmp