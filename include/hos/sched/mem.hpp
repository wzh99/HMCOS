#pragma once

#include <hos/util/util.hpp>
#include <hos/util/vec.hpp>

namespace hos {

class MemStateIter;

/// Vector of memory states
class MemStateVec {
public:
    MemStateVec(int64_t init = 0) : init(init) {}

    int64_t Latest() const { return stables.Empty() ? init : stables.Back(); }

    int64_t Peak() const {
        return transients.Empty() ? init : transients.Max();
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
        transients.Append(up);
        stables.Append(down);
    }

    /// Extend this state vector with the other vector
    /// State values in that vector will be offset by latest state of this
    /// vector
    void Extend(const MemStateVec& other);

    std::pair<int64_t, int64_t> operator[](size_t i) const {
        LOG_ASSERT(i < Size());
        return {transients[i], stables[i]};
    }

    size_t Size() const { return transients.Size(); }

    MemStateIter begin() const;
    MemStateIter end() const;

    const StatVec<int64_t>& Transients() const { return transients; }
    const StatVec<int64_t>& Stables() const { return stables; }

private:
    /// Initial memory offset
    int64_t init = 0;
    /// Transient states, when an op is being executed
    StatVec<int64_t> transients;
    /// Stable states, when execution of the op has been finished
    StatVec<int64_t> stables;
};

class MemStateIter {
public:
    using I64VecIter = std::vector<int64_t>::const_iterator;

    MemStateIter(I64VecIter transIter, I64VecIter stableIter)
        : t(transIter), s(stableIter) {}

    void operator++() {
        t++;
        s++;
    }

    std::pair<int64_t, int64_t> operator*() const { return {*t, *s}; }

    bool operator==(const MemStateIter& other) const { return t == other.t; }

    bool operator!=(const MemStateIter& other) const {
        return !operator==(other);
    }

private:
    I64VecIter t, s;
};

inline MemStateIter MemStateVec::begin() const {
    return {transients.begin(), stables.begin()};
}

inline MemStateIter MemStateVec::end() const {
    return {transients.end(), stables.end()};
}

inline void MemStateVec::Extend(const MemStateVec& other) {
    auto last = Latest();
    for (auto [t, s] : other) {
        transients.Append(t + last);
        stables.Append(s + last);
    }
}

}  // namespace hos