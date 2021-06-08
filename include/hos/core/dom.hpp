#pragma once

#include <hos/core/vertex.hpp>

namespace hos {

/// Node in dominator tree
template <class Vert>
class DomNode {
public:
    DomNode(const std::shared_ptr<Vert> &vertex) : vertex(vertex) {}

    bool Dominates(const DomNode &other) const {
        return this->in <= other.in && this->out >= other.out;
    }

    /// Point back to original vertex
    std::weak_ptr<Vert> vertex;
    /// Parent in dominator tree
    std::shared_ptr<DomNode> parent;
    /// Children in dominator tree
    std::vector<std::weak_ptr<DomNode>> children;

private:
    /// In and out index for O(1) time dominance decision
    uint32_t in = 0, out = 0;

    template <class>
    friend class NodeNumberer;
};

template <class Vert, class Ret, class... Args>
class DomTreeVisitor {
public:
    virtual Ret Visit(const std::shared_ptr<DomNode<Vert>> &node,
                      Args... args) = 0;
};

/// Node in depth-first spanning tree
/// This node is intermediate structure during construction of dominator tree.
template <class Vert>
struct DfNode {
    /// Node pointer is null
    static constexpr auto NONE = UINT32_MAX;

    /// Point to the actual vertex
    std::shared_ptr<Vert> vertex;
    /// Parent of this node in depth-first spanning tree
    uint32_t parent;
    /// Semi-dominator
    uint32_t semi;
    /// Set of nodes whose semi-dominator is this node
    std::vector<uint32_t> bucket;
    /// Immediate dominator
    uint32_t idom;
    /// Tree root in the forest
    uint32_t ancestor;
    /// Intermediate evaluation result
    uint32_t best;
    /// Size of the subtree with this node as root
    uint32_t size;
    /// Child of this node
    uint32_t child;
};

/// Builder of dominator tree, which implements Lengaur-Tarjan algorithm.
/// See https://www.cl.cam.ac.uk/~mr10/lengtarj.pdf for introduction of this
/// algorithm.
template <class Vert>
class DomBuilder {
public:
    using VertRef = std::shared_ptr<Vert>;
    using DomNodeRef = std::shared_ptr<DomNode<Vert>>;
    using VertListFunc = std::function<std::vector<VertRef>(const VertRef &)>;

    DomBuilder(VertListFunc getPreds = std::mem_fn(&Vert::Preds),
               VertListFunc getSuccs = std::mem_fn(&Vert::Succs))
        : getPreds(getPreds), getSuccs(getSuccs) {}

    std::vector<DomNodeRef> Build(const VertRef &root);

private:
    using DfNodeType = DfNode<Vert>;

    uint32_t eval(uint32_t v);
    void compress(uint32_t v);
    void link(uint32_t v, uint32_t w);

#define DFNODE_FIELD(name) \
    uint32_t &name(uint32_t v) { return nodes[v].name; }

    DFNODE_FIELD(parent)
    DFNODE_FIELD(semi)
    DFNODE_FIELD(idom)
    DFNODE_FIELD(ancestor)
    DFNODE_FIELD(best)
    DFNODE_FIELD(size)
    DFNODE_FIELD(child)

#undef DFNODE_FIELD

    VertListFunc getPreds, getSuccs;
    std::vector<DfNodeType> nodes;
    std::unordered_map<VertRef, uint32_t> vertIdx;
};

template <class Vert>
class NodeNumberer : public DomTreeVisitor<Vert, Unit> {
public:
    Unit Visit(const std::shared_ptr<DomNode<Vert>> &node) override {
        node->in = number++;
        for (auto &child : node->children) Visit(child.lock());
        node->out = number++;
        return {};
    }

private:
    uint32_t number = 0;
};

template <class Vert>
std::vector<std::shared_ptr<DomNode<Vert>>> DomBuilder<Vert>::Build(
    const std::shared_ptr<Vert> &root) {
    // Find all nodes by depth-first search
    LOG_ASSERT(root);
    DfsIter<Vert> end;
    uint32_t count = 0;
    for (auto it = DfsIter<Vert>({root}, getSuccs); it != end; ++it) {
        this->nodes.push_back({
            *it,               // vertex
            DfNodeType::NONE,  // parent
            DfNodeType::NONE,  // semi
            {},                // bucket
            DfNodeType::NONE,  // idom
            DfNodeType::NONE,  // ancestor
            count,             // best
            0,                 // size
            DfNodeType::NONE   // child
        });
        vertIdx.insert({*it, count});
        count++;
    }

    // Find parent for each child
    if (nodes.size() <= 1) {
        LOG(ERROR) << "Graph is trivial. No need to build dominator tree.";
        return {};
    }
    for (auto v = 0u; v < nodes.size(); v++) {
        semi(v) = v;
        for (auto &wVert : getSuccs(nodes[v].vertex)) {
            auto w = vertIdx[wVert];
            if (semi(w) == DfNodeType::NONE) parent(w) = v;
        }
    }

    for (auto w = uint32_t(nodes.size() - 1); w >= 1; w--) {
        // Compute semi-dominator of each vertex
        auto p = parent(w);
        for (auto &vVert : getPreds(nodes[w].vertex)) {
            auto v = vertIdx[vVert];
            auto u = eval(v);
            if (semi(w) > semi(u)) semi(w) = semi(u);
        }
        AddUnique(nodes[semi(w)].bucket, w);
        link(p, w);

        // Implicitly define immediate dominators
        for (auto v : nodes[p].bucket) {
            auto u = eval(v);
            idom(v) = semi(u) < semi(v) ? u : p;
        }
        nodes[p].bucket.clear();
    }

    // Explicitly define immediate dominators
    std::vector<DomNodeRef> results;
    for (auto &node : nodes)
        results.push_back(std::make_shared<DomNode<Vert>>(node.vertex));
    for (auto v = 1u; v < nodes.size(); v++) {
        if (idom(v) != semi(v)) idom(v) = idom(idom(v));
        auto d = idom(v);
        results[v]->parent = results[d];
        results[d]->children.push_back(results[v]);
    }

    // Number all nodes for O(1) dominance decision
    NodeNumberer<Vert>().Visit(results[0]);

    return results;
}

template <class Vert>
uint32_t DomBuilder<Vert>::eval(uint32_t v) {
    if (ancestor(v) == DfNodeType::NONE)
        return v;
    else {
        compress(v);
        auto b = best(v), a = ancestor(v), ba = best(a);
        return semi(ba) < semi(b) ? ba : b;
    }
}

template <class Vert>
void DomBuilder<Vert>::compress(uint32_t v) {
    auto a = ancestor(v);
    if (ancestor(a) == DfNodeType::NONE) return;
    compress(a);
    if (semi(best(a)) < semi(best(v))) best(v) = best(a);
    ancestor(v) = ancestor(a);
}

/// Add edge `(v, w)` to the forest
template <class Vert>
void DomBuilder<Vert>::link(uint32_t v, uint32_t w) {
    auto s = w;
    while (child(s) != DfNodeType::NONE &&
           semi(best(w)) < semi(best(child(s)))) {
        // Combine the first two trees in the child chain, making the larger one
        // the combined root.
        auto cs = child(s), ss = size(s);
        auto ccs = child(cs), scs = size(cs);
        auto sccs = size(ccs);
        if (ss + sccs >= 2 * scs) {
            ancestor(cs) = s;
            child(s) = ccs;
        } else {
            size(cs) = ss;
            ancestor(s) = cs;
            s = cs;
        }
    }

    // Combine the two forests giving the combination the child chain of the
    // smaller forest. The other child chain is then collapsed, giving all its
    // trees ancestor link to v.
    best(s) = best(w);
    if (size(v) < size(w)) std::swap(s, child(v));
    size(v) += size(w);
    while (s != DfNodeType::NONE) {
        ancestor(s) = v;
        s = child(s);
    }
}

}  // namespace hos