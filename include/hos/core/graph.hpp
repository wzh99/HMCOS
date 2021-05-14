#pragma once

#include <hos/core/value.hpp>
#include <hos/util/util.hpp>

namespace hos {

struct Vertex : public Object<Vertex> {
    static constexpr uint32_t typeIndex =
        (Object::typeIndex << BASE_INDEX_SHIFT) + 1;

    /// Predecessors and successors of vertex
    /// All elements in predecessor or successor list must be distinct.
    /// (Multi-edges are not allowed)
    std::vector<VertexRef> preds, succs;

    static void Connect(const VertexRef &tail, const VertexRef &head) {
        AddUnique(tail->succs, head);
        AddUnique(head->preds, tail);
    }
};

struct Input : public Object<Input>, public Vertex {
    static constexpr uint32_t typeIndex =
        (Vertex::typeIndex << BASE_INDEX_SHIFT) + 1;

    /// Value that this vertex corresponds to
    ValueRef value;

    Input(const ValueRef &val) : value(val) {
        LOG_ASSERT(val->kind == ValueKind::INPUT);
    }
};

struct Output : public Object<Output>, public Vertex {
    static constexpr uint32_t typeIndex =
        (Vertex::typeIndex << BASE_INDEX_SHIFT) + 2;

    /// Value that this vertex corresponds to
    ValueRef value;

    Output(const ValueRef &val) : value(val) {
        LOG_ASSERT(val->kind == ValueKind::RESULT);
    }
};

using OutputRef = std::shared_ptr<Output>;

struct Op : public Object<Op>, public Vertex {
    static constexpr uint32_t typeIndex =
        (Vertex::typeIndex << BASE_INDEX_SHIFT) + 3;

    /// `NodeProto` pointer in ONNX graph
    const onnx::NodeProto *node;
    /// Input and output values of this operator
    std::vector<ValueRef> inputs, outputs;
    /// Parent in dominance tree
    OpRef parent;
    /// Children in dominance tree
    std::vector<OpRef> children;

    Op(const onnx::NodeProto *node) : node(node) {}

    const std::string &GetName() const { return node->op_type(); }
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