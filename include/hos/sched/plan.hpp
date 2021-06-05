#pragma once

#include <hos/sched/life.hpp>
#include <optional>

namespace hos {

/// Spatial-temporal descriptor of a value in memory
struct MemoryDesc : public Lifetime {
    static constexpr uint64_t OFFSET_UNKNOWN = UINT64_MAX;

    /// Memory offset of this value
    uint64_t offset = OFFSET_UNKNOWN;
    /// Cached size of the value in bytes
    uint64_t size;

    explicit MemoryDesc(const Lifetime &life)
        : Lifetime(life), size(life.value->type.Size()) {}

    MemoryDesc(const MemoryDesc &desc) = default;

    std::string Format() const {
        return fmt::format("t[{}:{}] s[{}:{}] {}", gen, kill, offset,
                           offset + size, value->name);
    }
};

inline bool CmpBySizeRev(const MemoryDesc &lhs, const MemoryDesc &rhs) {
    return lhs.size > rhs.size;
}

/// Abstraction of the packing status in container
struct Step {
    /// Beginning time of this step
    int32_t begin;
    /// Time length that keeps offset of this step
    /// time + width must be time of next step, if it exists
    int32_t width;
    /// Memory offset
    uint64_t offset;

    int32_t End() const { return begin + width; }

    bool CanPlace(const MemoryDesc &desc) const {
        return begin <= desc.gen && End() >= desc.kill;
    }

    std::string Format() const {
        return fmt::format("{}:{}@{}", begin, End(), offset);
    }
};

inline bool CmpByBegin(const Step &lhs, const Step &rhs) {
    return lhs.begin < rhs.begin;
}

inline bool CmpByOffset(const Step &lhs, const Step &rhs) {
    return lhs.offset < rhs.offset;
}

/// Contains rectangular items
class Container {
public:
    Container(int32_t begin, int32_t end)
        : tBegin(begin), tEnd(end), maxHeight(0) {
        steps.push_back({begin, end - begin, 0});
    }

    template <class Cmp>
    const Step &FindMinBy(Cmp cmp) const {
        return MinElem(steps, cmp);
    }

    uint64_t GetMaxHeight() const { return maxHeight; }

    /// Place a memory block in container, return memory offset of the block.
    uint64_t Place(int32_t begin, int32_t width, uint64_t height);

    /// Lift one step to merge it with the neighbor with lowest offset
    void Lift(int32_t time);

    /// Print steps in container
    void Print() const {
        fmt::print("Steps: \n");
        for (auto &s : steps) fmt::print("{}\n", s.Format());
        fmt::print("\n");
    }

private:
    /// Find index of the step at given time
    int32_t findStepAt(int32_t time) const;
    /// Try a number of times to merge a step with its next
    void tryMerge(size_t beginIdx, uint32_t nTrial);

    /// Temporal range of this container
    int32_t tBegin, tEnd;
    /// Maximal height of this container
    uint64_t maxHeight;
    /// Steps in this container, sorted by time in increasing order
    std::vector<Step> steps;
};

template <class Elem, class Pred, class Cmp>
inline auto MinPosWithConstr(const std::vector<Elem> &vec, Pred pred, Cmp cmp) {
    std::optional<typename std::vector<Elem>::const_iterator> minPos;
    for (auto it = vec.begin(); it != vec.end(); it++) {
        if (!pred(*it)) continue;
        if (!minPos.has_value() || cmp(*it, *minPos.value())) minPos = it;
    }
    return minPos;
}

struct MemoryPlan {
    /// Peak memory footprint
    uint64_t peak;
    /// Spatial-temporal descriptor of values in memory
    std::vector<MemoryDesc> descs;
    /// Maps values to its offset
    std::unordered_map<ValueRef, uint64_t> valToOff;

    /// Create a memory plan with peak and meory descriptors
    MemoryPlan(uint64_t peak, std::vector<MemoryDesc> &&descs);
    /// Print memory plan
    void Print() const;
    /// Visualize memory plan with Matplotlib
    void Visualize(const std::string &dir, const std::string &name,
                   const std::string &format = "pdf");
};

/// Implement best-fit heuristic by Sekiyama et al.
MemoryPlan BestFit(const LifetimeStat &stat);

};  // namespace hos