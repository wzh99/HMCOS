#pragma once

#include <hos/util/util.hpp>
#include <hos/util/vec.hpp>

namespace hos {

class MemStateIter;

/// Vector of memory states
class MemStateVec {
public:
    MemStateVec(int64_t init = 0) : init(init) {}

    int64_t Latest() const { return transients.Empty() ? init : transients.Back(); }

    int64_t Peak() const {
        return stables.Empty() ? init : stables.Max();
    }

    std::pair<int64_t, int64_t> ComputeState(uint64_t inc, uint64_t dec) const {
        auto up = Latest() + inc;
        auto down = up - dec;
        return {up, down};
    }

    /// Append one state to vector with memory increase when transitioned to
    /// transient state and decrease when transitioned to stable state
    void Append(uint64_t inc, uint64_t dec) {
        auto [up, down] = ComputeState(inc, dec);
        stables.Append(up);
        transients.Append(down);
    }

    /// Extend this state vector with the other vector
    /// State values in that vector will be offset by latest state of this
    /// vector
    void Extend(const MemStateVec& other);

    void Swap(MemStateVec& other) {
        std::swap(this->init, other.init);
        this->stables.Swap(other.stables);
        this->transients.Swap(other.transients);
    }

    std::pair<int64_t, int64_t> operator[](size_t i) const {
        LOG_ASSERT(i < Size());
        return {stables[i], transients[i]};
    }

    size_t Size() const { return stables.Size(); }

    MemStateIter begin() const;
    MemStateIter end() const;

    const StatVec<int64_t>& Stables() const { return stables; }
    const StatVec<int64_t>& Transients() const { return transients; }

private:
    /// Initial memory offset
    int64_t init = 0;
    /// Stable states, when an op is being executed
    StatVec<int64_t> stables;
    /// Transient states, when execution of the op has been finished
    StatVec<int64_t> transients;
};

using I64VecConstIter = std::vector<int64_t>::const_iterator;

class MemStateIter : public ZipIter<I64VecConstIter, I64VecConstIter> {
public:
    MemStateIter(I64VecConstIter tIter, I64VecConstIter sIter)
        : ZipIter(tIter, sIter) {}
};

inline MemStateIter MemStateVec::begin() const {
    return {stables.begin(), transients.begin()};
}

inline MemStateIter MemStateVec::end() const {
    return {stables.end(), transients.end()};
}

inline void MemStateVec::Extend(const MemStateVec& other) {
    auto last = Latest();
    for (auto [t, s] : other) {
        stables.Append(t + last);
        transients.Append(s + last);
    }
}

}  // namespace hos