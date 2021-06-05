#pragma once

#include <glog/logging.h>

#include <algorithm>
#include <numeric>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

namespace hos {

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

template <class Iterable, class BinOp, class Lhs>
inline auto Accumulate(const Iterable &elems, BinOp binOp, Lhs init) {
    return std::accumulate(elems.begin(), elems.end(), init, binOp);
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

}  // namespace hos