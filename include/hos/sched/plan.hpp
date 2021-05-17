#pragma once

#include <hos/sched/life.hpp>

namespace hos {

struct MemoryPlan {
    /// Peak memory footprint
    uint64_t peak;
    /// Spatial-temporal descriptor of values in memory
    std::vector<MemoryDesc> descs;
    /// Maps values to its offset
    std::unordered_map<ValueRef, uint64_t> valToOff;
};

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

    std::string Format() const {
        return fmt::format("{}:{}@{}", begin, End(), offset);
    }
};

struct CmpByBegin {
    bool operator()(const Step &lhs, const Step &rhs) const {
        return lhs.begin < rhs.begin;
    }
};

/// Contains rectangular items
class Container {
public:
    Container(int32_t begin, int32_t end) : tBegin(begin), tEnd(end) {
        steps.push_back({begin, end - begin, 0});
    }

    template <class Cmp>
    const Step& FindMinBy(Cmp cmp) const {
        return std::min_element(steps.begin(), steps.end(), cmp);
    }

    /// Place a rectangular item in container, return whether the placement
    /// succeeded.
    bool Place(int32_t begin, int32_t width, uint64_t height);

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

/// Implement best-fit heuristic by Sekiyama et al.
MemoryPlan BestFit(const std::vector<Lifetime> &lifetimes);

};  // namespace hos