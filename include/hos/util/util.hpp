#pragma once

#include <glog/logging.h>

#include <iterator>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

namespace hos {

/// String Utility

template <class StrIterable>
inline std::string Join(const StrIterable &strs, std::string &&sep,
                        std::string &&prefix = "", std::string &&suffix = "") {
    // Return empty string if there are no elements
    if (strs.empty()) return prefix + suffix;

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
                                 std::string &&prefix = "",
                                 std::string &&suffix = "") {
    return Join(strs, ", ", std::move(prefix), std::move(suffix));
}

/// Functional Utility

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

/// Vector Utility

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

template <class Elem, class Cmp>
inline const Elem &ReduceMin(const std::vector<Elem> &vec, Cmp cmp) {
    LOG_ASSERT(!vec.empty());
    return *std::min_element(vec.begin(), vec.end(), cmp);
}

/// Set Utility

template <class KeyType, class ValueType>
inline bool Contains(const std::unordered_set<KeyType, ValueType> &map,
                     const KeyType &elem) {
    return map.find(elem) != map.end();
}

/// Map Utility

template <class KeyType, class ValueType>
inline bool Contains(const std::unordered_map<KeyType, ValueType> &map,
                     const KeyType &elem) {
    return map.find(elem) != map.end();
}

/// ONNX Utility

}  // namespace hos