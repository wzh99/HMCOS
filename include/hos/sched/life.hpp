#pragma once

#include <hos/sched/sched.hpp>

namespace hos {

/// Lifetime descriptor of a value during computation
struct Lifetime {
    /// Value that this struct describes
    ValueRef value;
    /// Lifetime of a value is an interval [gen, kill). `gen` and `kill` are
    /// all indices of ops.
    int32_t gen, kill;

    /// Input time when no computation has been done
    static constexpr int32_t TIME_INPUT = -1;
    /// Unknown time
    static constexpr int32_t TIME_UNKNOWN = INT32_MAX;

    int32_t Length() const { return kill - gen; }

    void Print() const {
        LOG(INFO) << gen << ':' << kill << ' ' << value->name;
    }
};

inline bool CmpByGenKill(const Lifetime &lhs, const Lifetime &rhs) {
    if (lhs.gen != rhs.gen) return lhs.gen < rhs.gen;
    return lhs.kill < rhs.kill;
}

inline bool CmpByLength(const Lifetime &lhs, const Lifetime &rhs) {
    auto ll = lhs.Length(), rl = rhs.Length();
    return ll != rl ? ll < rl : CmpByGenKill(lhs, rhs);
}

inline bool CmpByLengthRev(const Lifetime &lhs, const Lifetime &rhs) {
    return CmpByLength(rhs, lhs);
}

class UsageIter;

/// Lifetime statistics of all values in a computation graph
struct LifetimeStat {
    /// Lifetime range of values
    std::pair<int32_t, int32_t> range;
    /// Lifetimes of each value
    std::vector<Lifetime> values;

    UsageIter begin() const;
    UsageIter end() const;
};

class UsageIter {
public:
    UsageIter(int32_t t, const std::vector<Lifetime> &values)
        : t(t), values(values) {}

    uint64_t operator*();
    void operator++() { t++; }

    bool operator!=(const UsageIter &other) const { return this->t != other.t; }

private:
    int32_t t;
    size_t idx = 0;
    uint64_t sum = 0;
    const std::vector<Lifetime> &values;
    std::vector<const Lifetime *> alive;
};

inline UsageIter LifetimeStat::begin() const { return {range.first, values}; }
inline UsageIter LifetimeStat::end() const { return {range.second, values}; }

/// Whether the only output of this op can overlap one of the input
uint32_t OverlapInput(const OpRef &op);
static constexpr auto OVERLAP_FAILED = UINT32_MAX;

/// Compute lifetime statistics of a complete op sequence of a graph.
LifetimeStat ComputeLifetime(const std::vector<OpRef> &opSeq,
                             const Graph &graph);

/// Estimate peak memory usage of an op sequence. This sequence does not need to
/// contain all the ops in the graph.
uint64_t EstimatePeak(const std::vector<OpRef> &seq,
                      const std::vector<InputRef> &inputs);

}  // namespace hos
