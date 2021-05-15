#pragma once

#include <fmt/core.h>
#include <glog/logging.h>

#include <cstdint>
#include <functional>
#include <utility>

namespace hos {

template <class Derived, class Base>
bool Is(const std::shared_ptr<Base> &ptr) {
    return ptr->GetKind() == Derived::classKind;
}

template <class Derived, class Base>
std::shared_ptr<Derived> As(const std::shared_ptr<Base> &ptr) {
    if (!Is<Derived>(ptr))
        LOG(FATAL) << fmt::format("Object is not of type `{}`.",
                                  typeid(Derived).name());
    else
        return std::shared_ptr<Derived>(
            ptr, static_cast<Derived *>(ptr.get()));
}

}  // namespace hos
