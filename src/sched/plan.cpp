#include <hos/sched/plan.hpp>

namespace hos {

bool Container::Place(int32_t begin, int32_t width, uint64_t height) {
    // Find the step at `begin`
    auto end = begin + width;
    if (begin < tBegin || end >= tEnd) {
        LOG(ERROR) << fmt::format("Expect begin in [{}, {}), got {}.", tBegin,
                                  tEnd - width, begin);
        return false;
    }
    auto idx = findStepAt(begin);
    auto step = steps[idx];

    // Check if the item can be placed at this step
    if (end > step.End()) {
        LOG(ERROR) << fmt::format(
            "Item beginning at {} of width {} cannot be placed at step {}",
            begin, width, step.Format());
        return false;
    }

    // Update maximal height
    auto newHeight = step.offset + height;
    maxHeight = std::max(maxHeight, newHeight);

    // Place this item by modifying current steps
    steps.erase(steps.begin() + idx);
    std::vector<Step> inserted;
    if (begin != step.begin)  // gap on left fringe
        inserted.push_back({step.begin, begin - step.begin, step.offset});
    inserted.push_back({begin, width, newHeight});
    if (end != step.End())  // gap on right fringe
        inserted.push_back({end, step.End() - end, step.offset});
    steps.insert(steps.begin() + idx, inserted.begin(), inserted.end());

    /// Merge steps that have same offsets to one step
    auto beginIdx = std::max(idx - 1, 0);
    tryMerge(beginIdx, uint32_t(inserted.size() + 1));

    return true;
}

void Container::Lift(int32_t time) {
    // Find step
    auto idx = findStepAt(time);
    auto &step = steps[idx];

    // Lift step to its lowest neighbor
    auto leftIdx = std::max<int32_t>(idx - 1, 0);
    auto rightIdx = std::min<int32_t>(idx + 1, int32_t(steps.size()) - 1);
    auto lOff = steps[leftIdx].offset, rOff = steps[rightIdx].offset;
    if (lOff < step.offset || rOff < step.offset) {
        LOG(ERROR) << fmt::format(
            "Step {} is not lower than both of its neighbors.", step.Format());
        return;
    }
    step.offset = std::min(lOff, rOff);

    // Merge steps with same offsets
    tryMerge(leftIdx, 3);
}

inline int32_t Container::findStepAt(int32_t time) const {
    LOG_ASSERT(time >= tBegin && time < tEnd);
    auto next = std::upper_bound(steps.begin(), steps.end(), Step{time, 0, 0},
                                 CmpByBegin());
    return int32_t(next - steps.begin()) - 1;
}

void Container::tryMerge(size_t beginIdx, uint32_t nTrial) {
    auto stepIdx = beginIdx;
    for (auto i = 0u; i < nTrial; i++) {
        // No steps behind, cannot merge further
        if (stepIdx == steps.size() - 1) return;

        // Merge two steps if they have same offsets
        auto curStep = &steps[stepIdx], nextStep = &steps[stepIdx + 1];
        if (curStep->offset == nextStep->offset) {
            curStep->width += nextStep->width;
            steps.erase(steps.begin() + stepIdx + 1);
        } else
            stepIdx++;
    }
}

inline static std::vector<MemoryDesc> ltToDesc(
    const std::vector<Lifetime> &lifetimes) {
    return Transform<std::vector<MemoryDesc> >(
        lifetimes, [](auto &lt) { return MemoryDesc(lt); });
}

MemoryPlan BestFit(const std::vector<Lifetime> &lifetimes) {
    // Initialize memory descriptors
    auto descs = ltToDesc(lifetimes);

    return MemoryPlan();
}

}  // namespace hos