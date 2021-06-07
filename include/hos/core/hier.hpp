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

    virtual std::string Format() const = 0;
    virtual HierKind GetKind() const = 0;
};

using HierVertRef = std::shared_ptr<HierVertex>;
using HierDomNodeRef = std::shared_ptr<DomNode<HierVertex>>;

/// Equivalent to `Input`, but appear in a hierarchical graph.
struct HierInput : public HierVertex {
    ValueRef value;

    HierInput(const ValueRef &val) : value(val) {
        LOG_ASSERT(val->kind == ValueKind::INPUT);
    }

    std::string Format() const override { return value->name; }

    static constexpr auto classKind = HierKind::INPUT;
    HierKind GetKind() const override { return classKind; }
};

using HierInputRef = std::shared_ptr<HierInput>;

/// Equivalent to `Output`, but appear in a hierarchical graph.
struct HierOutput : public HierVertex {
    ValueRef value;

    HierOutput(const ValueRef &val) : value(val) {
        LOG_ASSERT(val->kind == ValueKind::RESULT);
    }

    std::string Format() const override { return value->name; }

    static constexpr auto classKind = HierKind::OUTPUT;
    HierKind GetKind() const override { return classKind; }
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
    /// There are some differences between inputs of an op and those of a
    /// sequence:
    /// 1. Parameters are not considered inputs in the sequence.
    /// 2. All input values in sequence must be unique.
    std::vector<ValueRef> inputs, outputs;
    /// Group that this sequence reside in
    std::weak_ptr<Group> group;

    Sequence(const OpRef &op);

    OpRef Entrance() const { return ops.front(); }
    OpRef Exit() const { return ops.back(); }

    std::string Format() const override;

    static constexpr auto classKind = HierKind::SEQUENCE;
    HierKind GetKind() const override { return classKind; }
};

using SequenceRef = std::shared_ptr<Sequence>;

/// A group of sequences
struct Group : public HierVertex {
    /// All sequences in this group
    std::vector<SequenceRef> seqs;
    /// Entrance and exit sequences of this group
    /// Entrance sequences have no predecessors, and exit sequences must have no
    /// successors.
    std::vector<SequenceRef> entrs, exits;
    /// Input and output values of this group
    /// Inputs must be union of inputs of all entrance sequences.
    /// Outputs must be union of outputs of all exit sequences.
    std::vector<ValueRef> inputs, outputs;

    static constexpr auto classKind = HierKind::GROUP;
    HierKind GetKind() const override { return classKind; }
};

using GroupRef = std::shared_ptr<Group>;

/// A hierarchical graph, created from normal graph
struct HierGraph {
    // Original computation graph
    const Graph &graph;
    /// Only inputs and outputs are explicited stored, and others are connected
    /// by references and can be found through traversal of the graph.
    std::vector<HierInputRef> inputs;
    std::vector<HierOutputRef> outputs;
    /// Maps vertices in normal graph to ones in hierarchical graph
    std::unordered_map<VertexRef, HierVertRef> vertMap;

    explicit HierGraph(const Graph &graph);

    /// Visualize all levels of structures in this hierarchical graph
    void VisualizeAll(const std::string &dir, const std::string &name,
                      const std::string &format = "pdf");

    /// Visualize top level vertices in this graph
    void VisualizeTop(const std::string &dir, const std::string &name,
                      const std::string &format = "pdf");

    /// Visualize dominator (or post-dominator) tree of this hierarchical graph
    void VisualizeDom(bool post, const std::string &dir,
                      const std::string &name,
                      const std::string &format = "pdf");
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
        switch (vert->GetKind()) {
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