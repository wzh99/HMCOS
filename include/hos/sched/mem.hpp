#pragma once

#include <hos/util/util.hpp>
#include <hos/util/vec.hpp>

namespace hos {

/// A sequence of memory states
class MemStateList {
public:
    std::pair<int64_t, int64_t> ComputeState(uint64_t inc, uint64_t dec) const {
        auto up = latest + inc;
        auto down = up - dec;
        return {up, down};
    }

    void Append(uint64_t inc, uint64_t dec) {
        auto [up, down] = ComputeState(inc, dec);
        transients.Append(up);
        stables.Append(down);
        latest = down;
    }

    std::pair<int64_t, int64_t> operator[](size_t i) const {
        LOG_ASSERT(i < transients.Size());
        return {transients[i], stables[i]};
    }

    const StatVec<int64_t>& Transients() const { return transients; }
    const StatVec<int64_t>& Stables() const { return stables; }

private:
    /// Latest stable memory
    int64_t latest = 0;
    /// Transient states, when an op is being executed
    StatVec<int64_t> transients;
    /// Stable states, when execution of the op has been finished
    StatVec<int64_t> stables;
};

}  // namespace hos