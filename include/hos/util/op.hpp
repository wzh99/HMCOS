#pragma once

#include <hos/util/util.hpp>

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

bool IsElementWise(const std::string &name);

}