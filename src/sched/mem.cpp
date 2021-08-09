#include <hos/sched/life.hpp>
#include <hos/sched/mem.hpp>

namespace hos {

std::pair<uint64_t, uint64_t> ComputeIncDec(
    const OpRef &op, const std::vector<ValueRef> &killed) {
    // See if output value can overlap one of the input
    auto ovlIdx = OverlapInput(op);
    if (ovlIdx != OVERLAP_FAILED && !Contains(killed, op->inputs[ovlIdx]))
        ovlIdx = OVERLAP_FAILED;

    // Compute increase in size at transition to transient state
    uint64_t inc = 0;
    if (ovlIdx == OVERLAP_FAILED)
        inc = std::transform_reduce(op->outputs.begin(), op->outputs.end(),
                                    0ull, std::plus(),
                                    [](auto &val) { return val->type.Size(); });

    // Compute decrease in size at transition to stable state
    auto ovlVal = ovlIdx == OVERLAP_FAILED ? nullptr : op->inputs[ovlIdx];
    auto dec = 0ull;
    for (auto &val : op->inputs) {
        if (val->kind == ValueKind::PARAM) continue;  // skip parameters
        if (!Contains(killed, val)) continue;         // value is not killed
        if (val == ovlVal) continue;  // overlapped value should not be counted
        dec += val->type.Size();
    }

    return {inc, dec};
}

}  // namespace hos
