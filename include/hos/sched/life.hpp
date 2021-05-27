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

    void Print() const { fmt::print("{}:{} {}\n", gen, kill, value->name); }
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

/// Lifetime statistics of all values in a computation graph
class LifetimeStat {
public:
    /// Lifetime limit of values
    int32_t begin, end;
    /// Lifetimes of each value
    std::vector<Lifetime> blocks;

    /// Get memory histogram in [begin, end)
    std::vector<uint64_t> Histogram() const;
    /// Get peak memory usage
    uint64_t Peak() const;

private:
    /// Count memory usage in [begin, end)
    void count(std::function<void(uint64_t)> callback) const;
};

LifetimeStat ComputeLifetime(const OpSeq &opSeq, const Graph &graph);

}  // namespace hos