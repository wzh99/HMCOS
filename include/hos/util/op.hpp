#pragma once

#include <onnx/defs/schema.h>

namespace hos {

enum OpTrait {
    NONE = 0,
    ELEMENT_WISE = 1 << 0,
};

class OpTraitRegistry {
public:
    static void Init();
    static bool Match(const std::string &name, OpTrait trait);

private:
    static std::unordered_map<std::string, OpTrait> opTraits;
};

}