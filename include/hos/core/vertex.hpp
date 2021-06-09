#pragma once

#include <hos/util/util.hpp>

namespace hos {
template <class Vert>
struct VertexBase {
    using VertRef = std::shared_ptr<Vert>;
    using VertWeakRef = std::weak_ptr<Vert>;

    /// Predecessor list of vertex
    /// All elements in predecessor or successor list must be distinct.
    /// (Multi-edges are not allowed)
    std::vector<VertWeakRef> preds;
    std::vector<VertRef> succs;

    std::vector<VertRef> Preds() const {
        return Transform<std::vector<VertRef>>(
            preds, [](const VertWeakRef &v) { return v.lock(); });
    }

    std::vector<VertRef> Succs() const { return succs; }

    static void Connect(const VertRef &tail, const VertRef &head) {
        AddUnique(tail->succs, head);
        AddUnique(head->preds, VertWeakRef(tail));
    }

    static void Disconnect(const VertRef &tail, const VertRef &head) {
        Remove(tail->succs, head);
        Remove(head->preds, VertWeakRef(tail));
    }

    static void ReplaceSuccOfPred(const VertRef &pred, const VertRef &oldVert,
                                  const VertRef &newVert) {
        if (Contains(pred->succs, newVert))
            Remove(pred->succs, oldVert);
        else
            std::replace(pred->succs.begin(), pred->succs.end(), oldVert,
                         newVert);
    }

    static void ReplaceSuccOfAllPreds(const VertRef &oldVert,
                                      const VertRef &newVert) {
        for (auto &pred : oldVert->preds)
            ReplaceSuccOfPred(pred.lock(), oldVert, newVert);
    }

    static void ReplacePredOfSucc(const VertRef &succ, const VertRef &oldVert,
                                  const VertRef &newVert) {
        if (std::find_if(succ->preds.begin(), succ->preds.end(), [&](auto &v) {
                return v.lock() == newVert;
            }) != succ->preds.end())
            RemoveIf(succ->preds, [&](auto &v) { return v.lock() == oldVert; });
        else
            std::replace_if(
                succ->preds.begin(), succ->preds.end(),
                [&](auto &v) { return v.lock() == oldVert; },
                VertWeakRef(newVert));
    }

    static void ReplacePredOfAllSuccs(const VertRef &oldVert,
                                      const VertRef &newVert) {
        for (auto &succ : oldVert->succs)
            ReplacePredOfSucc(succ, oldVert, newVert);
    }

    static void Replace(const VertRef &oldVert, const VertRef &newVert) {
        ReplaceSuccOfAllPreds(oldVert, newVert);
        ReplacePredOfAllSuccs(oldVert, newVert);
    }
};

template <class Vert, class Iter>
class VertIter {
public:
    using VertRef = std::shared_ptr<Vert>;

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
    bool hasTraversed(const VertRef &v) const { return Contains(traversed, v); }

private:
    VertRef next;
    std::unordered_set<VertRef> traversed;
};

template <class Vert>
class DfsIter : public VertIter<Vert, DfsIter<Vert>> {
public:
    using VertRef = std::shared_ptr<Vert>;
    using VertListFunc = std::function<std::vector<VertRef>(const VertRef &)>;

    DfsIter() = default;
    DfsIter(const std::vector<VertRef> &inputs,
            VertListFunc getSuccs = std::mem_fn(&Vert::Succs));

    bool End() const { return stack.empty(); }
    VertRef Loop();

private:
    VertListFunc getSuccs;
    std::vector<VertRef> stack;
};

template <class Vert>
DfsIter<Vert>::DfsIter(const std::vector<VertRef> &inputs,
                       VertListFunc getSuccs)
    : getSuccs(getSuccs) {
    for (auto it = inputs.rbegin(); it != inputs.rend(); it++)
        stack.push_back(*it);
    this->operator++();
}

template <class Vert>
std::shared_ptr<Vert> DfsIter<Vert>::Loop() {
    // Pop one vertex
    auto vertex = stack.back();
    stack.pop_back();

    // Skip travered vertex
    if (this->hasTraversed(vertex)) return nullptr;

    // Add successors to stack
    auto succs = getSuccs(vertex);
    for (auto it = succs.rbegin(); it != succs.rend(); it++)
        stack.push_back(*it);

    return vertex;
}

template <class Vert>
class RpoIter : public VertIter<Vert, RpoIter<Vert>> {
public:
    using VertRef = std::shared_ptr<Vert>;
    using VertListFunc = std::function<std::vector<VertRef>(const VertRef &)>;

    RpoIter() = default;
    RpoIter(const std::vector<VertRef> &outputs,
            VertListFunc getPreds = std::mem_fn(&Vert::Preds));

    bool End() const { return stack.empty(); }
    VertRef Loop();

private:
    struct StackRecord {
        VertRef vertex;
        bool visited;
    };

    VertListFunc getPreds;
    std::vector<StackRecord> stack;
};

template <class Vert>
RpoIter<Vert>::RpoIter(const std::vector<std::shared_ptr<Vert>> &outputs,
                       VertListFunc getPreds)
    : getPreds(getPreds) {
    for (auto it = outputs.rbegin(); it != outputs.rend(); it++)
        stack.push_back({*it, false});
    this->operator++();
}

template <class Vert>
std::shared_ptr<Vert> RpoIter<Vert>::Loop() {
    // Pop one vertex
    auto [vertex, visited] = stack.back();
    stack.pop_back();

    // Skip if this vertex is traversed before
    if (this->hasTraversed(vertex)) return nullptr;

    // Apply function to vertex if it has been visited
    if (visited) return vertex;

    // Otherwise add predecessors to stack
    stack.push_back({vertex, true});
    auto preds = getPreds(vertex);
    for (auto it = preds.rbegin(); it != preds.rend(); it++)
        stack.push_back({*it, false});

    return nullptr;
}

template <class Vert, class Iter>
class VertRange {
public:
    using VertRef = std::shared_ptr<Vert>;

    VertRange(const std::vector<VertRef> &init) : init(init) {}

    Iter begin() const { return Iter(init); }
    Iter end() const { return Iter(); }

private:
    std::vector<VertRef> init;
};

}  // namespace hos