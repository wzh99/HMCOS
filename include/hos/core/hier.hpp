#pragma once

#include <hos/core/dom.hpp>
#include <hos/core/graph.hpp>

namespace hos {

enum class HierKind {
    INPUT,
    OUTPUT,
    SEQUENCE,
    GROUP,
};

struct HierVertex : public VertexBase<HierVertex> {
    /// Node of this vertex in dominator and post-dominator tree
    std::shared_ptr<DomNode<HierVertex>> dom, postDom;

    bool Dominates(const HierVertex &other, bool strict = false) const {
        return this->dom->Dominates(*other.dom, strict);
    }

    bool PostDominates(const HierVertex &other, bool strict = false) const {
        return this->postDom->Dominates(*other.postDom, strict);
    }

    /// Label of this vertex in visualization
    virtual std::string Label() const = 0;
    /// Vertex kind for RTTI
    virtual HierKind Kind() const = 0;
};

using HierVertRef = std::shared_ptr<HierVertex>;
using HierVertWeakRef = std::weak_ptr<HierVertex>;
using HierDomNodeRef = std::shared_ptr<DomNode<HierVertex>>;

/// Equivalent to `Input`, but appear in a hierarchical graph.
struct HierInput : public HierVertex {
    ValueRef value;

    HierInput(const ValueRef &val) : value(val) {
        LOG_ASSERT(val->kind == ValueKind::INPUT);
    }

    std::string Label() const override { return value->name; }

    static constexpr auto classKind = HierKind::INPUT;
    HierKind Kind() const override { return classKind; }
};

using HierInputRef = std::shared_ptr<HierInput>;

/// Equivalent to `Output`, but appear in a hierarchical graph.
struct HierOutput : public HierVertex {
    ValueRef value;

    HierOutput(const ValueRef &val) : value(val) {
        LOG_ASSERT(val->kind == ValueKind::RESULT);
    }

    std::string Label() const override { return value->name; }

    static constexpr auto classKind = HierKind::OUTPUT;
    HierKind Kind() const override { return classKind; }
};

using HierOutputRef = std::shared_ptr<HierOutput>;

struct Group;

/// A sequence of ops
/// All ops, except the first one, must only consume values produced by op in
/// front of it.
/// All ops, except the last one, must produce values that are all consumed by
/// op next to it.
struct Sequence : public HierVertex {
    /// All ops in this sequence
    std::vector<OpRef> ops;
    /// Input and output values of this sequence
    /// Parameters are not considered inputs in sequences.
    std::vector<ValueRef> inputs, outputs;
    /// Group where this sequence resides in
    std::weak_ptr<Group> group;

    explicit Sequence(const OpRef &op);
    std::string Label() const override;

    bool Contains(const OpRef &op) const { return hos::Contains(ops, op); }

    static constexpr auto classKind = HierKind::SEQUENCE;
    HierKind Kind() const override { return classKind; }
};

using SequenceRef = std::shared_ptr<Sequence>;

/// A group of sequences
struct Group : public HierVertex {
    /// All sequences in this group
    std::vector<SequenceRef> seqs;
    /// Entrance and exit sequences of this group
    /// Predecessors of each entrance must all be outside of the group.
    /// Successors of each exits must all be outside of the group
    std::vector<SequenceRef> entrs, exits;
    /// In and out frontiers of the this group
    /// Each input frontier must have at least one predecessor from sequence
    /// outside the group. Each output frontier must have at least one successor
    /// from sequence outside the group
    std::vector<SequenceRef> inFront, outFront;
    /// Use count of input and output values
    /// Here we adopt producer-consumer model to describe def-use chains. When a
    /// value is defined, it produces a number of use counts. When it is used,
    /// it consumes one use count. Only def-use chains across groups are
    /// counted.
    std::vector<std::pair<ValueRef, uint32_t>> consumed, produced;

    std::string Label() const override;

    bool Contains(const SequenceRef &seq) const {
        return seq->group.lock().get() == this;
    }

    bool Contains(const OpRef &op) const {
        return std::any_of(seqs.begin(), seqs.end(),
                           [&](auto &seq) { return seq->Contains(op); });
    }

    template <class Vert>
    bool Contains(const HierVertRef &vert) const {
        return Is<Vert>(vert) && this->Contains(Cast<Vert>(vert));
    }

    static constexpr auto classKind = HierKind::GROUP;
    HierKind Kind() const override { return classKind; }
};

using GroupRef = std::shared_ptr<Group>;

/// A hierarchical graph, created from normal graph
struct HierGraph {
    // Original computation graph
    const Graph &graph;
    /// Only inputs and outputs are explicited stored, and others are connected
    /// by edges and can be found through traversal of the graph.
    std::vector<HierInputRef> inputs;
    std::vector<HierOutputRef> outputs;

    explicit HierGraph(const Graph &graph);

    /// Visualize all levels of structures in this hierarchical graph
    void VisualizeAll(const std::string &dir, const std::string &name,
                      const std::string &format = "pdf");

    /// Visualize top level vertices in this hierarchical graph
    void VisualizeTop(const std::string &dir, const std::string &name,
                      const std::string &format = "pdf");

    /// Visualize dominator tree of this hierarchical graph
    void VisualizeDom(const std::string &dir, const std::string &name,
                      const std::string &format = "pdf");

    /// Visualize post-dominator tree of this hierarchical graph
    void VisualizePostDom(const std::string &dir, const std::string &name,
                          const std::string &format = "pdf");
};

class RpoHierRange : public VertRange<HierVertex, RpoIter<HierVertex>> {
public:
    RpoHierRange(const HierGraph &hier)
        : VertRange(Transform<std::vector<HierVertRef>>(
              hier.outputs, [](auto &out) { return HierVertRef(out); })) {}
};

/// Visitor of vertices in hierarchical graph
template <class Ret, class... Args>
class HierVertVisitor {
public:
    virtual Ret VisitInput(const HierInputRef &input, Args... args) = 0;
    virtual Ret VisitOutput(const HierOutputRef &output, Args... args) = 0;
    virtual Ret VisitSequence(const SequenceRef &seq, Args... args) = 0;
    virtual Ret VisitGroup(const GroupRef &group, Args... args) = 0;

    virtual Ret Visit(const HierVertRef &vert, Args... args) {
        if (Contains(memo, vert)) return memo[vert];
        Ret ret;
        switch (vert->Kind()) {
            case HierKind::INPUT:
                ret = VisitInput(Cast<HierInput>(vert),
                                 std::forward<Args>(args)...);
                break;
            case HierKind::OUTPUT:
                ret = VisitOutput(Cast<HierOutput>(vert),
                                  std::forward<Args>(args)...);
                break;
            case HierKind::SEQUENCE:
                ret = VisitSequence(Cast<Sequence>(vert),
                                    std::forward<Args>(args)...);
                break;
            case HierKind::GROUP:
                ret =
                    VisitGroup(Cast<Group>(vert), std::forward<Args>(args)...);
                break;
            default:
                LOG(FATAL) << "Unreachable.";
        }
        memo.insert({vert, ret});
        return ret;
    }

protected:
    std::unordered_map<HierVertRef, Ret> memo;
};

/// Interface for passes on hierarchical graphs
class HierGraphPass {
public:
    /// Run the pass on a hierarchical graph.
    /// The pass will mutate the graph instead producing a new graph.
    virtual void Run(HierGraph &graph) = 0;
};

/// Run a series of passes with static polymorphism
template <class... Passes>
inline void RunPass(HierGraph &graph) {
    (void)std::initializer_list<int>{0, (Passes().Run(graph), 0)...};
}

}  // namespace hos