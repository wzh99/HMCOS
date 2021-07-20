#pragma once

#include <fmt/format.h>

#include <chrono>

namespace hos {

void PrintProgress(size_t index, size_t size,
                   std::chrono::system_clock::time_point start);

class ProgressIter {
public:
    using Clock = std::chrono::system_clock;

    ProgressIter(size_t index, size_t size) : index(index), size(size) {}

    void operator++() { PrintProgress(++index, size, start); }

    size_t operator*() const { return index; }

    bool operator!=(const ProgressIter &other) const {
        return this->index != other.index;
    }

private:
    size_t index;
    size_t size;
    Clock::time_point start = Clock::now();
};

class ProgressRange {
public:
    ProgressRange(size_t size) : size(size) {}

    ProgressIter begin() const {
        PrintProgress(0, size, {});
        return ProgressIter(0, size);
    }

    ProgressIter end() const { return ProgressIter(size, size); }

    ~ProgressRange() { printf("\n"); }

private:
    size_t size;
};

}  // namespace hos