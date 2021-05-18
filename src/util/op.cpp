#include <fmt/core.h>
#include <glog/logging.h>

#include <hos/util/op.hpp>
#include <hos/util/util.hpp>

namespace hos {

std::unordered_set<std::string> ewOps{
    "Abs",        "Add",   "And",   "Neg",         "Mul",
    "Exp",        "Div",   "Ceil",  "Not",         "LeakyRelu",
    "Elu",        "Equal", "Floor", "Greater",     "HardSigmoid",
    "Selu",       "Less",  "PRelu", "Log",         "Or",
    "Reciprocal", "Pow",   "Relu",  "Sigmoid",     "Softplus",
    "Softsign",   "Sqrt",  "Sub",   "Tanh",        "Xor",
    "Acos",       "Asin",  "Atan",  "Cos",         "Sin",
    "Tan",        "Sinh",  "Cosh",  "Asinh",       "Acosh",
    "Atanh",      "Sign",  "Erf",   "Mod",         "ThresholdedRelu",
    "BitShift",   "Round", "Celu",  "LessOrEqual", "GreaterOrEqual",
    "HardSwish",  "Clip"};

std::unordered_map<std::string, OpTrait> OpTraitRegistry::opTraits;

static OpTrait extractTrait(const onnx::OpSchema &schema) {
    OpTrait trait = NONE;
    std::string doc(schema.doc());

    // Element-wise
    auto npos = std::string::npos;
    if (Contains(ewOps, schema.Name()))
        trait = OpTrait(trait | OpTrait::ELEMENT_WISE);

    return trait;
}

void OpTraitRegistry::Init() {
    for (auto &schema : onnx::OpSchemaRegistry::get_all_schemas())
        opTraits.insert({schema.Name(), extractTrait(schema)});
}

bool OpTraitRegistry::Match(const std::string &name, OpTrait trait) {
    if (!Contains(opTraits, name)) {
        LOG(ERROR) << fmt::format("Op '{}' not found in registry.", name);
        return false;
    }
    return (opTraits[name] & trait) == trait;
}

}  // namespace hos
