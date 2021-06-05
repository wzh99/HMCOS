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

}