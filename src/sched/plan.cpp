#include <filesystem>
#include <fstream>
#include <hos/sched/plan.hpp>
#include <hos/util/viz.hpp>

namespace hos {

uint64_t Container::Place(int32_t begin, int32_t width, uint64_t height) {
    // Find the step at `begin`
    auto end = begin + width;
    if (begin < tBegin || end > tEnd) {
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
        return MemoryDesc::OFFSET_UNKNOWN;
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

    return step.offset;
}

void Container::Lift(int32_t time) {
    // Do nothing if there is only one step
    if (steps.size() == 1) {
        LOG(ERROR) << "There is only one step in container. Cannot lift.";
        return;
    }

    // Find step
    auto idx = findStepAt(time);
    auto &step = steps[idx];

    // Lift step to its lowest neighbor
    if (idx == 0) {
        auto &rightStep = steps[1];
        if (step.offset > rightStep.offset) {
            LOG(ERROR) << fmt::format("Step {} is higher than step {}.",
                                      step.Format(), rightStep.Format());
            return;
        }
        step.offset = rightStep.offset;
        tryMerge(idx, 1);
    } else if (idx == int32_t(steps.size()) - 1) {
        auto &leftStep = steps[idx - 1];
        if (step.offset > leftStep.offset) {
            LOG(ERROR) << fmt::format("Step {} is higher than step {}.",
                                      step.Format(), leftStep.Format());
            return;
        }
        step.offset = leftStep.offset;
        tryMerge(idx - 1, 1);
    } else {
        auto &leftStep = steps[idx - 1], &rightStep = steps[idx + 1];
        if (step.offset > leftStep.offset || step.offset > rightStep.offset) {
            LOG(ERROR) << fmt::format(
                "Step {} is higher than step {} or step {}.", step.Format(),
                leftStep.Format(), rightStep.Format());
            return;
        }
        step.offset = std::min(leftStep.offset, rightStep.offset);
        tryMerge(idx - 1, 2);
    }
}

inline int32_t Container::findStepAt(int32_t time) const {
    LOG_ASSERT(time >= tBegin && time < tEnd);
    auto next = std::upper_bound(steps.begin(), steps.end(), Step{time, 0, 0},
                                 CmpByBegin);
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

MemoryPlan::MemoryPlan(uint64_t peak, std::vector<MemoryDesc> &&descs)
    : peak(peak), descs(std::move(descs)) {
    // Sort memory descriptors according to lifetime
    std::sort(this->descs.begin(), this->descs.end(), CmpByGenKill);

    // Map values to offsets
    for (auto &desc : descs) valToOff.insert({desc.value, desc.offset});
}

void MemoryPlan::Print() const {
    fmt::print("Peak: {}\n", peak);
    fmt::print("\nPlan: \n");
    for (auto &desc : descs) fmt::print("{}\n", desc.Format());
}

void MemoryPlan::Visualize(const std::string &dir, const std::string &name,
                           const std::string &format) {
    RectPlot plot(name);
    for (auto &desc : descs)
        plot.AddRect(float(desc.gen), float(desc.offset), float(desc.Length()),
                     float(desc.size));
    plot.Render(dir, format);
}

inline static std::vector<MemoryDesc> ltToDesc(
    const std::vector<Lifetime> &lifetimes) {
    return Transform<std::vector<MemoryDesc> >(
        lifetimes, [](auto &lt) { return MemoryDesc(lt); });
}

MemoryPlan BestFit(const LifetimeStat &stat) {
    // Initialize unplaced memory descriptors and container
    auto unplaced = ltToDesc(stat.values);
    Container cont(stat.range.first, stat.range.second);

    // Iterate until no blocks remain
    std::vector<MemoryDesc> placed;
    while (!unplaced.empty()) {
        // Choose step with lowest offset
        auto &step = cont.FindMinBy(CmpByOffset);

        // Find best fit for this step
        auto bestFitPos = MinPosWithConstr(
            unplaced, [&](auto &desc) { return step.CanPlace(desc); },
            CmpByLengthRev);

        // Lift this step if no block can be placed
        if (!bestFitPos.has_value()) {
            cont.Lift(step.begin);
            continue;
        }

        // Place best fit block at the step
        auto block = *bestFitPos.value();
        block.offset = cont.Place(block.gen, block.Length(), block.size);
        placed.push_back(block);
        unplaced.erase(bestFitPos.value());
    }

    return MemoryPlan(cont.GetMaxHeight(), std::move(placed));
}

}  // namespace hos