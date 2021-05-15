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

    VertexKind GetKind() const { return VertexKind::INPUT; }
};

struct Output : public Vertex {
    /// Value that this vertex corresponds to
    ValueRef value;

    Output(const ValueRef &val) : value(val) {
        LOG_ASSERT(val->kind == ValueKind::RESULT);
    }

    static constexpr auto classKind = VertexKind::OUTPUT;

    VertexKind GetKind() const { return VertexKind::OUTPUT; }
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

    VertexKind GetKind() const { return VertexKind::OP; }
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
    void Traverse(const std::function<void(VertexRef)> &func);

    /// Visualize vertices and edges in the graph.
    /// Vertices are connected according to their def-use relations. Values will
    /// not appear in the visualization.
    void Visualize(const std::string &dir,
                   const std::string &format = "pdf") const;

private:
    /// Owns the ONNX model so that all pointers or references to objects in it
    /// remain valid
    std::unique_ptr<onnx::ModelProto> model;
};

using GraphRef = std::shared_ptr<Graph>;

}  // namespace hos