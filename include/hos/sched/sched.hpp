#pragma once

#include <hos/core/graph.hpp>

namespace hos {

struct Lifetime {
    /// Lifetime of a value is an interval [begin, end). `begin` and `end` are
    /// all indices of ops.
    int32_t begin, end;

    /// Input time when no computation has been done
    static constexpr int32_t INPUT = -1;
    /// Unknown time
    static constexpr int32_t UNKNOWN = INT32_MAX;

    int32_t Length() const {
        LOG_ASSERT(begin < end);
        return end - begin;
    }

    bool operator<(const Lifetime &other) const {
        if (this->begin != other.begin) return this->begin < other.begin;
        return this->end < other.end;
    }
};

struct OpSched {
    /// Op sequence to be executed
    std::vector<OpRef> opSeq;
    /// Litetime of each value defined by ops
    std::unordered_map<ValueRef, Lifetime> valLife;

    OpSched(const std::vector<OpRef> &opSeq, const Graph &graph);

    void Print() const;
};

}  // namespace hos