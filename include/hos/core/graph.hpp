#pragma once

#include <hos/core/value.hpp>
#include <hos/util/util.hpp>

namespace hos {

struct Vertex {
    /// Predecessors and successors of vertex
    /// All elements in predecessor or successor list must be distinct.
    /// (Multi-edges are not allowed)
    std::vector<VertexRef> preds, succs;

    static void Connect(const VertexRef &tail, const VertexRef &head) {
        AddUnique(tail->succs, head);
        AddUnique(head->preds, tail);
    }

    static void Disconnect(const VertexRef &tail, const VertexRef &head) {
        Remove(tail->succs, head);
        Remove(head->preds, tail);
    }

    enum class VertexKind {
        INPUT,
        OUTPUT,
        OP,
    };

    virtual VertexKind GetKind() const = 0;

protected:
    Vertex() = default;
};

struct Input : public Vertex {
    /// Value that this vertex corresponds to
    ValueRef value;

    Input(const ValueRef &val) : value(val) {
        LOG_ASSERT(val->kind == ValueKind::INPUT);
    }

    static constexpr auto classKind = VertexKind::INPUT;

    VertexKind GetKind() const override { return VertexKind::INPUT; }
};

struct Output : public Vertex {
    /// Value that this vertex corresponds to
    ValueRef value;

    Output(const ValueRef &val) : value(val) {
        LOG_ASSERT(val->kind == ValueKind::RESULT);
    }

    static constexpr auto classKind = VertexKind::OUTPUT;

    VertexKind GetKind() const override { return VertexKind::OUTPUT; }
};

using OutputRef = std::shared_ptr<Output>;

struct Op : Vertex {
    /// `NodeProto` pointer in ONNX graph
    const onnx::NodeProto *node;
    /// Input and output values of this operator
    std::vector<ValueRef> inputs, outputs;

    Op(const onnx::NodeProto *node) : node(node) {}

    const std::string &GetType() const { return node->op_type(); }

    static constexpr auto classKind = VertexKind::OP;

    VertexKind GetKind() const override { return VertexKind::OP; }
};

class Graph {
public:
    // Name of this graph
    std::string name;
    /// Input vertices of the graph
    std::vector<InputRef> inputs;
    /// Output vertices of the graph
    std::vector<OutputRef> outputs;
    /// Parameters of graph
    std::vector<ValueRef> params;
    /// All operators in graph
    std::vector<OpRef> ops;

    /// Create a graph from ONNX model.
    /// Note that the constructor assumes that all intermediates in model are
    /// type-checked and their types are stored in `value_info` field of graph.
    /// Otherwise the constructor will panic.
    Graph(std::unique_ptr<onnx::ModelProto> &&model,
          const std::string &name = "");

    /// Traverse the graph in reverse post-order.
    /// This method is suitable for simple traversal that does not depend on
    /// results of predecessors. If results of predecessors are needed, consider
    /// using `GraphVisitor`.
    void Traverse(const std::function<void(const VertexRef &)> &func) const;

    /// Visualize vertices and edges in the graph.
    /// Vertices are connected according to their def-use relations. Values will
    /// not appear in the visualization.
    void Visualize(const std::string &dir,
                   const std::string &format = "pdf") const;

private:
    void connectVerts();

    /// Owns the ONNX model so that all pointers or references to objects in it
    /// remain valid
    std::unique_ptr<onnx::ModelProto> model;
};

template <class Ret, class... Args>
class GraphVisitor {
public:
    virtual Ret Visit(const VertexRef &vert, Args... args) {
        using Kind = Vertex::VertexKind;

        if (Contains(memo, vert)) return memo[vert];

        Ret ret;
        switch (vert->GetKind()) {
            case Kind::INPUT:
                ret = VisitInput(As<Input>(vert), std::forward<Args>(args)...);
                break;
            case Kind::OUTPUT:
                ret =
                    VisitOutput(As<Output>(vert), std::forward<Args>(args)...);
                break;
            case Kind::OP:
                ret = VisitOp(As<Op>(vert), std::forward<Args>(args)...);
                break;
            default:
                LOG(FATAL) << "Unreachable.";
        }
        return ret;
    }

    virtual Ret VisitInput(const InputRef &input, Args... args) = 0;
    virtual Ret VisitOutput(const OutputRef &output, Args... args) = 0;
    virtual Ret VisitOp(const OpRef &op, Args... args) = 0;

protected:
    std::unordered_map<VertexRef, Ret> memo;
};

using GraphRef = std::shared_ptr<Graph>;

}  // namespace hos