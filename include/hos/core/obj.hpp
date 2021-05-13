#pragma once

#include <fmt/core.h>
#include <glog/logging.h>

#include <cstdint>
#include <functional>
#include <utility>

namespace hos {

/// Object interface for RTTI using CRTP
/// If one wants to use this interface, `Object` must appear BEFORE `ObjectBase`
/// or other subclasses of `ObjectBase` in the inheritance list.
template <class ObjType>
class Object {
public:
    /// All classes that derive from `Object` has to declare static constexpr
    /// member `typeIndex`. Different classes must have different indices.
    static constexpr uint32_t typeIndex = 0;

    virtual uint32_t GetTypeIndex() { return ObjType::typeIndex; }

    template <class TargetType>
    bool Is() {
        if constexpr (std::is_same<ObjType, TargetType>::value)
            return true;
        else if constexpr (std::is_base_of<TargetType, ObjType>::value)
            return true;
        else
            return this->GetTypeIndex() == TargetType::typeIndex;
    }
};

/// Type index of a derived class is shift of base class plus its own offset.
static constexpr uint32_t BASE_INDEX_SHIFT = 4;

template <class ObjType, class TargetObjType>
std::shared_ptr<TargetObjType> As(const std::shared_ptr<ObjType> &ptr) {
    if (!ptr->Is<TargetObjType>())
        LOG(FATAL) << fmt::format("Object is not of type `{}`.",
                                  typeid(TargetObjType).name());
    else
        return std::shared_ptr<TargetObjType>(
            ptr, static_cast<TargetObjType *>(ptr.get()));
}

}  // namespace hos
