#pragma once

#include <hos/util/util.hpp>

namespace hos {
template <class VertType>
struct AbstractVertex {
    using VertRef = std::shared_ptr<VertType>;
    using VertWeakRef = std::weak_ptr<VertType>;

    /// Predecessor list of vertex
    /// All elements in predecessor or successor list must be distinct.
    /// (Multi-edges are not allowed)
    std::vector<VertWeakRef> preds;
    std::vector<VertRef> succs;

    std::vector<VertRef> GetPreds() const {
        return Transform<std::vector<VertRef>>(
            preds, [](const VertWeakRef &v) { return v.lock(); });
    }

    std::vector<VertRef> GetSuccs() const { return succs; }

    static void Connect(const VertRef &tail, const VertRef &head) {
        AddUnique(tail->succs, head);
        AddUnique(head->preds, VertWeakRef(tail));
    }

    static void Disconnect(const VertRef &tail, const VertRef &head) {
        Remove(tail->succs, head);
        Remove(head->preds, VertWeakRef(tail));
    }

    static void Replace(const VertRef &oldVert, const VertRef &newVert) {
        for (auto &predWeak : oldVert->preds) {
            auto pred = predWeak.lock();
            std::replace(pred->succs.begin(), pred->succs.end(), oldVert,
                         newVert);
        }
        for (auto &succ : oldVert->succs)
            std::replace_if(
                succ->preds.begin(), succ->preds.end(),
                [&](const VertWeakRef &v) { return v.lock() == oldVert; },
                VertWeakRef(newVert));
    }
};

template <class VertType, class Iter>
class VertIter {
public:
    using VertRef = std::shared_ptr<VertType>;

    bool End() const { LOG(FATAL) << "End() not implemented."; }
    VertRef Loop() { LOG(FATAL) << "Loop() not implemented."; }

    void operator++() {
        auto iter = static_cast<Iter *>(this);
        while (!iter->End()) {
            auto result = iter->Loop();
            if (result) {
                this->next = result;
                this->traversed.insert(result);
                return;
            }
        }
        this->next = nullptr;
    }

    VertRef operator*() const { return next; }

    bool operator==(const VertIter &other) const {
        return this->next == other.next;
    }

    bool operator!=(const VertIter &other) const {
        return !this->operator==(other);
    }

protected:
    bool hasTraversed(const VertRef &v) const {
        return Contains(traversed, v);
    }

private:
    VertRef next;
    std::unordered_set<VertRef> traversed;
};

template <class VertType>
class DfsIter : public VertIter<VertType, DfsIter<VertType>> {
public:
    using VertRef = std::shared_ptr<VertType>;

    DfsIter() = default;
    DfsIter(const std::vector<VertRef> &inputs);

    bool End() const { return stack.empty(); }
    VertRef Loop();

private:
    std::vector<VertRef> stack;
};

template <class VertType>
DfsIter<VertType>::DfsIter(
    const std::vector<std::shared_ptr<VertType>> &inputs) {
    for (auto it = inputs.rbegin(); it != inputs.rend(); it++)
        stack.push_back(*it);
    this->operator++();
}

template <class VertType>
typename std::shared_ptr<VertType> DfsIter<VertType>::Loop() {
    // Pop one vertex
    auto vertex = stack.back();
    stack.pop_back();

    // Skip travered vertex
    if (this->hasTraversed(vertex)) return nullptr;

    // Add successors to stack
    auto succs = vertex->GetSuccs();
    for (auto it = succs.rbegin(); it != succs.rend(); it++)
        stack.push_back(*it);

    return vertex;
}

template <class VertType>
class RpoIter : public VertIter<VertType, RpoIter<VertType>> {
public:
    using VertRef = std::shared_ptr<VertType>;

    RpoIter() = default;
    RpoIter(const std::vector<VertRef> &outputs);

    bool End() const { return stack.empty(); }
    VertRef Loop();

private:
    struct StackRecord {
        VertRef vertex;
        bool visited;
    };

    std::vector<StackRecord> stack;
};

template <class VertType>
RpoIter<VertType>::RpoIter(
    const std::vector<std::shared_ptr<VertType>> &outputs) {
    for (auto it = outputs.rbegin(); it != outputs.rend(); it++)
        stack.push_back({*it, false});
    this->operator++();
}

template <class VertType>
typename std::shared_ptr<VertType> RpoIter<VertType>::Loop() {
    // Pop one vertex
    auto [vertex, visited] = stack.back();
    stack.pop_back();

    // Skip if this vertex is traversed before
    if (this->hasTraversed(vertex)) return nullptr;

    // Apply function to vertex if it has been visited
    if (visited) return vertex;

    // Otherwise add predecessors to stack
    stack.push_back({vertex, true});
    auto preds = vertex->GetPreds();
    for (auto it = preds.rbegin(); it != preds.rend(); it++)
        stack.push_back({*it, false});

    return nullptr;
}

template <class VertType, class Iter>
class VertRange {
public:
    using VertRef = std::shared_ptr<VertType>;

    VertRange(const std::vector<VertRef> &init) : init(init) {}

    Iter begin() const { return Iter(init); }
    Iter end() const { return Iter(); }

private:
    std::vector<VertRef> init;
};

}  // namespace hos