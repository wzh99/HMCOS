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
    if (lhs.gen < rhs.gen) return lhs.gen < rhs.gen;
    return lhs.kill < rhs.kill;
}

inline bool CmpByLength(const Lifetime &lhs, const Lifetime &rhs) {
    auto ll = lhs.Length(), rl = rhs.Length();
    return ll != rl ? ll < rl : CmpByGenKill(lhs, rhs);
}

/// Lifetime statistics of all values in a computation graph
struct LifetimeStat {
    /// Lifetime limit of values
    int32_t begin, end;
    /// Lifetimes of each value
    std::vector<Lifetime> values;
};

LifetimeStat ComputeLifetime(const OpSeq &opSeq, const Graph &graph);

}  // namespace hos