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

struct CmpByGenKill {
    bool operator()(const Lifetime &lhs, const Lifetime &rhs) const {
        if (lhs.gen < rhs.gen) return lhs.gen < rhs.gen;
        return lhs.kill < rhs.kill;
    }
};

struct CmpByLength {
    bool operator()(const Lifetime &lhs, const Lifetime &rhs) const {
        auto ll = lhs.Length(), rl = rhs.Length();
        return ll != rl ? ll < rl : CmpByGenKill()(lhs, rhs);
    }
};

std::vector<Lifetime> ComputeLifetime(const OpSeq &opSeq, const Graph &graph);

/// Spatial-temporal descriptor of a value in memory
struct MemoryDesc : public Lifetime {
    static constexpr uint64_t OFFSET_UNKNOWN = UINT64_MAX;

    /// Memory offset of this value
    uint64_t offset = OFFSET_UNKNOWN;
    /// Cached size of the value in bytes
    uint64_t size;

    explicit MemoryDesc(const Lifetime &life)
        : Lifetime(life), size(life.value->type.Size()) {}

    void Print() const {
        fmt::print("t[{}:{}] s[{}:{}] {}\n", gen, kill, offset, size, value->name);
    }
};

}  // namespace hos