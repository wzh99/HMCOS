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

template <class VertType>
class RpoIter {
public:
    using VertRef = std::shared_ptr<VertType>;

    static RpoIter End() { return RpoIter(); }

    RpoIter(const std::vector<VertRef> &outputs);
    void operator++();

    VertRef operator*() const { return next; }

    bool operator==(const RpoIter &other) const {
        return this->next == other.next;
    }

    bool operator!=(const RpoIter &other) const {
        return !this->operator==(other);
    }

private:
    RpoIter() = default;

    struct StackRecord {
        VertRef vertex;
        bool visited;
    };

    VertRef next;
    std::vector<StackRecord> stack;
    std::unordered_set<VertRef> traversed;
};

template <class VertType>
RpoIter<VertType>::RpoIter(
    const std::vector<RpoIter<VertType>::VertRef> &outputs) {
    for (auto iter = outputs.rbegin(); iter != outputs.rend(); iter++)
        stack.push_back({*iter, false});
    this->operator++();
}

template <class VertType>
void RpoIter<VertType>::operator++() {
    while (!stack.empty()) {
        // Pop one vertex
        auto [vertex, visited] = stack.back();
        stack.pop_back();

        // Skip if this vertex is traversed before
        if (Contains(traversed, vertex)) continue;

        // Apply function to vertex if it has been visited
        if (visited) {
            this->next = vertex;
            traversed.insert(vertex);
            return;
        }

        // Otherwise add predecessors to stack
        stack.push_back({vertex, true});
        auto &preds = vertex->preds;
        for (auto iter = preds.rbegin(); iter != preds.rend(); iter++)
            stack.push_back({iter->lock(), false});
    }

    this->next = nullptr;
}

}  // namespace hos