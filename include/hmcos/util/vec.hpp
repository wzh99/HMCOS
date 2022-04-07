#pragma once

#include <limits>
#include <vector>

namespace hmcos {

/// Compute statistics with the vector
template <class Elem>
class StatVec {
public:
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

    void Swap(StatVec &other) {
        this->vec.swap(other.vec);
        std::swap(this->min, other.min);
        std::swap(this->max, other.max);
    }

    Elem operator[](size_t i) const { return vec[i]; }

    auto begin() const { return vec.begin(); }
    auto end() const { return vec.end(); }

private:
    std::vector<Elem> vec;
    Elem min = std::numeric_limits<Elem>::max();
    Elem max = std::numeric_limits<Elem>::min();
};

}  // namespace hmcos