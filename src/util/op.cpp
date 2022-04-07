#include <hmcos/util/op.hpp>

namespace hmcos {

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

bool IsElementWise(const std::string &name) { return Contains(ewOps, name); }

std::unordered_set<std::string> reinterpOps{"Squeeze", "Unsqueeze", "Reshape",
                                            "Flatten"};

bool IsReinterpret(const std::string &name) {
    return Contains(reinterpOps, name);
}

}  // namespace hmcos
