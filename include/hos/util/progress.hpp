#pragma once

#include <fmt/format.h>

#include <chrono>

namespace hos {

void PrintProgress(size_t index, size_t size,
                   std::chrono::system_clock::time_point start);

template <bool display>
class ProgressIter {
public:
    using Clock = std::chrono::system_clock;

    ProgressIter(size_t index, size_t size) : index(index), size(size) {}

    void operator++() {
        index++;
        if constexpr (display) PrintProgress(index, size, start);
    }

    size_t operator*() const { return index; }

    bool operator!=(const ProgressIter &other) const {
        return this->index != other.index;
    }

private:
    size_t index;
    size_t size;
    Clock::time_point start = Clock::now();
};

template <bool display = true>
class ProgressRange {
public:
    ProgressRange(size_t size) : size(size) {}

    auto begin() const {
        PrintProgress(0, size, {});
        return ProgressIter<display>(0, size);
    }

    auto end() const { return ProgressIter<display>(size, size); }

    ~ProgressRange() { if constexpr (display) printf("\n"); }

private:
    size_t size;
};

}  // namespace hos