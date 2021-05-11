#pragma once

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

/// Map Utility

template <class Map, class Elem>
inline bool Contains(const Map &map, const Elem &elem) {
    return map.find(elem) != map.end();
}

/// ONNX/Protobuf Utility

}  // namespace hos