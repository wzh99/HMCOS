#pragma once

#include <glog/logging.h>

#include <cstdint>
#include <utility>

class ObjectBase {
public:
    virtual ~ObjectBase() = default;

protected:
    ObjectBase() = default;

private:
    uint32_t refCnt = 0;

    template <class>
    friend class Ref;
};

template <class ObjType>
class Object {
public:
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

    Object() = default;
    Object(const Object& obj) = delete;
    Object(Object&& obj) = delete;
};

template <class ObjType>
class Ref {
public:
    template <class... Args>
    static Ref<ObjType> Make(Args&&... args) {
        return Ref<ObjType>(new ObjType(std::forward<Args>(args)...));
    }

    template <class OtherObjType>
    Ref(const Ref<OtherObjType>& ref) : Ref(ref.ptr) {}

    ~Ref() { decRef(); }

    ObjType* Get() const { return ptr; }

    void Swap(Ref<ObjType>& other) { std::swap(this->ptr, other->ptr); }

    template <class TargetObjType>
    Ref<TargetObjType> As() const {
        if (!Get()->Is<TargetObjType>())
            LOG(FATAL) << fmt::format("Object is not of type `{}`.",
                                      typeid(TargetObjType).name());
        else
            return Ref<TargetObjType>(static_cast<TargetObjType*>(ptr));
    }

    ObjType* operator->() const { return Get(); }

    ObjType& operator*() const { return *Get(); }

    template <class OtherObjType>
    Ref<ObjType>& operator=(const Ref<OtherObjType>& other) {
        decRef();
        ptr = other.ptr;
        incRef();
        return *this;
    }

    bool operator==(const Ref<ObjType>& other) {
        return this->ptr == other->ptr;
    }

    bool operator!=(const Ref<ObjType>& other) {
        return this->ptr != other->ptr;
    }

private:
    Ref(ObjType* ptr) : ptr(ptr) { incRef(); }

    void incRef() { ptr->refCnt++; }

    void decRef() {
        ptr->refCnt--;
        if (ptr->refCnt == 0) {
            delete ptr;
            ptr = nullptr;
        }
    }

    ObjType* ptr;

    template <class>
    friend class Ref;
};

class Vertex : public Object<Vertex>, public ObjectBase {
public:
    Vertex(uint32_t index) : ObjectBase(), index(index) {}

    static constexpr uint32_t typeIndex = 1;

private:
    uint32_t index;
};

class Op : public Object<Op>, public Vertex {
public:
    Op() : Vertex(0) {}
    static constexpr uint32_t typeIndex = 2;
};
