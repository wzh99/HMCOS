#pragma once

#include <hos/core/value.hpp>

namespace hos {

struct Vertex;
using VertexRef = std::shared_ptr<Vertex>;

struct Vertex : public Object<Vertex> {
    static constexpr uint32_t typeIndex =
        (Object::typeIndex << BASE_INDEX_SHIFT) + 1;

    /// Predecessors and successors of vertex
    std::vector<VertexRef> preds, succs;
};

struct Input : public Object<Input>, public Vertex {
    static constexpr uint32_t typeIndex =
        (Vertex::typeIndex << BASE_INDEX_SHIFT) + 1;

    /// Value that this vertex corresponds to
    ValueRef value;

    explicit Input(const ValueRef &val) : value(val) {
        LOG_ASSERT(val->kind == ValueKind::INPUT);
    }
};

using InputRef = std::shared_ptr<Input>;

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
    /// Parent and children in dominance tree
    OpRef parent;
    std::vector<OpRef> children;

    const std::string &GetName() const { return node->name(); }
};

class Graph {
public:
    /// Create a graph from ONNX model.
    /// Note that the constructor assumes that all intermediates in model are
    /// type-checked and their types are stored in `value_info` field of graph.
    /// Otherwise the constructor will panic.
    Graph(std::unique_ptr<onnx::ModelProto> &&model);

    /// Input vertices of the graph
    std::vector<InputRef> inputs;
    /// Output vertices of the graph
    std::vector<OutputRef> outputs;
    /// Parameters of graph
    std::vector<ValueRef> params;
    /// All operators in graph
    std::vector<OpRef> ops;

private:
    /// Owns the ONNX model so that all pointers or references to objects in it
    /// remain valid
    std::unique_ptr<onnx::ModelProto> model;
};

using GraphRef = std::shared_ptr<Graph>;

}  // namespace hos