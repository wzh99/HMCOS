#pragma once

#include <glog/logging.h>

#include <algorithm>
#include <functional>
#include <numeric>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

namespace hmcos {

/// Type definition

using Unit = std::monostate;

/// Smart pointer

template <class Elem>
inline bool operator==(const std::weak_ptr<Elem> &lhs,
                       const std::weak_ptr<Elem> &rhs) {
    if (lhs.expired() || rhs.expired()) return false;
    return lhs.lock() == rhs.lock();
}

/// String

template <class StrIterable>
inline std::string Join(const StrIterable &strs, const char *sep,
                        const char *prefix = "", const char *suffix = "") {
    // Return empty string if there are no elements
    if (strs.empty()) return std::string(prefix) + suffix;

    // Join strings
    auto it = strs.begin();
    std::stringstream ss;
    ss << prefix << *(it++);
    while (it != strs.end()) ss << sep << *(it++);
    ss << suffix;

    return ss.str();
}

template <class StrIterable>
inline std::string JoinWithComma(const StrIterable &strs,
                                 const char *prefix = "",
                                 const char *suffix = "") {
    return Join(strs, ", ", prefix, suffix);
}

/// Functional

template <class Dst, class Src, class F>
inline auto Transform(const Src &src, F func) {
    Dst dst;
    dst.reserve(src.size());
    std::transform(src.begin(), src.end(), std::back_inserter(dst), func);
    return dst;
}

template <class Dst, class Src, class Pred>
inline auto Filter(const Src &src, Pred pred) {
    Dst dst;
    std::copy_if(src.begin(), src.end(), std::back_inserter(dst), pred);
    return dst;
}

/// Vector

template <class Elem>
inline bool Contains(const std::vector<Elem> &vec, const Elem &elem) {
    return std::find(vec.begin(), vec.end(), elem) != vec.end();
}

template <class Elem>
inline void AddUnique(std::vector<Elem> &vec, const Elem &elem) {
    if (Contains(vec, elem)) return;
    vec.push_back(elem);
}

template <class Elem>
inline void Remove(std::vector<Elem> &vec, const Elem &val) {
    vec.erase(std::remove(vec.begin(), vec.end(), val), vec.end());
}

template <class Elem, class Pred>
inline void RemoveIf(std::vector<Elem> &vec, Pred pred) {
    vec.erase(std::remove_if(vec.begin(), vec.end(), pred), vec.end());
}

template <class Elem, class Cmp>
inline const Elem &MinElem(const std::vector<Elem> &vec, Cmp cmp) {
    LOG_ASSERT(!vec.empty());
    return *std::min_element(vec.begin(), vec.end(), cmp);
}

template <class Elem>
inline auto Insert(std::vector<Elem> &vec, const Elem &elem) {
    auto pos = std::upper_bound(vec.begin(), vec.end(), elem);
    auto idx = pos - vec.begin();
    vec.insert(pos, elem);
    return idx;
}

template <class Elem>
inline void Extend(std::vector<Elem> &lhs, const std::vector<Elem> &rhs) {
    lhs.insert(lhs.end(), rhs.begin(), rhs.end());
}

/// Set

template <class KeyType, class ValueType>
inline bool Contains(const std::unordered_set<KeyType, ValueType> &map,
                     const KeyType &elem) {
    return map.find(elem) != map.end();
}

/// Map

template <class KeyType, class ValueType>
inline bool Contains(const std::unordered_map<KeyType, ValueType> &map,
                     const KeyType &elem) {
    return map.find(elem) != map.end();
}

/// Iterator

template <class LhsIter, class RhsIter>
class ZipIter {
public:
    ZipIter(LhsIter lhs, RhsIter rhs) : lhs(lhs), rhs(rhs) {}

    void operator++() { ++lhs, ++rhs; }

    auto operator*() const {
        return std::pair<decltype(*lhs), decltype(*rhs)>(*lhs, *rhs);
    }

    bool operator!=(const ZipIter &other) const {
        return lhs != other.lhs && rhs != other.rhs;
    }

private:
    LhsIter lhs;
    RhsIter rhs;
};

template <class LhsCont, class RhsCont>
class ZipRange {
public:
    ZipRange(LhsCont &lhs, RhsCont &rhs) : lhs(lhs), rhs(rhs) {}

    auto begin() { return ZipIter(lhs.begin(), rhs.begin()); }
    auto end() { return ZipIter(lhs.end(), rhs.end()); }

private:
    LhsCont &lhs;
    RhsCont &rhs;
};

template <class Iter>
class EnumIter {
public:
    EnumIter(Iter iter, size_t index) : iter(iter), index(index) {}

    void operator++() { ++iter, ++index; }

    auto operator*() const {
        return std::pair<size_t, decltype(*iter)>(index, *iter);
    }

    bool operator!=(const EnumIter &other) const {
        return this->iter != other.iter;
    }

private:
    Iter iter;
    size_t index;
};

template <class Cont>
class EnumRange {
public:
    EnumRange(Cont &cont) : cont(cont) {}

    auto begin() { return EnumIter(cont.begin(), 0); }
    auto end() { return EnumIter(cont.end(), 0); }

private:
    Cont &cont;
};

/// Hash

template <class Elem>
inline size_t Hash(const Elem &elem) {
    return std::hash<Elem>()(elem);
}

template <class Elem>
inline size_t HashCombine(size_t seed, const Elem &elem) {
    return seed ^ (Hash(elem) + 0x9e3779b9 + (seed << 6) + (seed >> 2));
}

template <class Elem, class... Args>
inline size_t HashCombine(size_t seed, const Elem &elem, const Args &...args) {
    return HashCombine(HashCombine(seed, elem), args...);
}

template <class... Args>
inline size_t Hash(const Args &...args) {
    return HashCombine(0ull, args...);
}

}  // namespace hmcos

namespace std {

template <class Elem>
struct hash<std::vector<Elem>> {
    std::size_t operator()(const std::vector<Elem> &vec) const {
        auto seed = vec.size();
        for (auto &elem : vec) seed = hmcos::HashCombine(seed, elem);
        return seed;
    }
};

}  // namespace std