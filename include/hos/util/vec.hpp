#pragma once

#include <limits>
#include <vector>

namespace hos {

/// Compute statistics with the vector
template <class Elem>
class StatVec {
public:
    StatVec() = default;

    StatVec(const std::vector<Elem> &vec) : vec(vec) {
        if (vec.empty()) return;
        auto [minPos, maxPos] = std::minmax_element(vec.begin(), vec.end());
        min = *minPos;
        max = *maxPos;
    }

    void Append(const Elem &elem) {
        vec.push_back(elem);
        min = std::min(min, elem);
        max = std::max(max, elem);
    }

    size_t Size() const { return vec.size(); }
    bool Empty() const { return vec.empty(); }

    Elem Min() const { return min; }
    Elem Max() const { return max; }

    Elem Back() const { return vec.back(); }

    Elem operator[](size_t i) const { return vec[i]; }

    auto begin() const { return vec.begin(); }
    auto end() const { return vec.end(); }

private:
    std::vector<Elem> vec;
    Elem min = std::numeric_limits<Elem>::max();
    Elem max = std::numeric_limits<Elem>::min();
};

}  // namespace hos