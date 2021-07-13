#pragma once

#include <hos/core/value.hpp>
#include <hos/core/vertex.hpp>
#include <hos/util/util.hpp>

namespace hos {

enum class VertexKind {
    INPUT,
    OUTPUT,
    OP,
};

struct Vertex : public VertexBase<Vertex> {
    virtual VertexKind Kind() const = 0;
};

struct Input : public Vertex {
    /// Value that this vertex corresponds to
    ValueRef value;

    Input(const ValueRef &val) : value(val) {
        LOG_ASSERT(val->kind == ValueKind::INPUT);
    }

    static constexpr auto classKind = VertexKind::INPUT;
    VertexKind Kind() const override { return VertexKind::INPUT; }
};

using InputRef = std::shared_ptr<Input>;

struct Output : public Vertex {
    /// Value that this vertex corresponds to
    ValueRef value;

    Output(const ValueRef &val) : value(val) {
        LOG_ASSERT(val->kind == ValueKind::RESULT);
    }

    VertexRef Def() const { return preds[0].lock(); }

    static constexpr auto classKind = VertexKind::OUTPUT;
    VertexKind Kind() const override { return VertexKind::OUTPUT; }
};

using OutputRef = std::shared_ptr<Output>;

struct Op : Vertex {
    /// Name of this op
    std::string name;
    /// Type name of this op
    std::string type;
    /// Input and output values of this operator
    std::vector<ValueRef> inputs, outputs;

    Op(const onnx::NodeProto *node)
        : name(node->name()), type(node->op_type()) {}

    Op(const Op &other) : name(other.name), type(other.type) {}

    static constexpr auto classKind = VertexKind::OP;
    VertexKind Kind() const override { return VertexKind::OP; }
};

using OpRef = std::shared_ptr<Op>;

struct Graph {
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

    Graph() = default;

    /// Create a graph from ONNX model.
    /// Note that the constructor assumes that all intermediates in model are
    /// type-checked and their types are stored in `value_info` field of graph.
    /// Otherwise the constructor will panic.
    Graph(const onnx::ModelProto &model, const std::string &name = "");

    /// Clone this graph.
    /// All vertices and values in this graph will be cloned, not
    /// reference-copied.
    Graph Clone() const;

    /// Extract a subgraph of this graph.
    Graph Subgraph(std::function<bool(const OpRef &)> isOutput,
                   const std::string &subName) const;

    /// Plot vertices and edges in the graph.
    /// Vertices are connected according to their def-use relations. Values will
    /// not appear in the visualization.
    void Plot(const std::string &dir,
                   const std::string &format = "pdf") const;

    /// Connect vertices according to their def-use relations
    void ConnectVerts();
};

class RpoVertRange : public VertRange<Vertex, RpoIter<Vertex>> {
public:
    RpoVertRange(const Graph &graph)
        : VertRange(Transform<std::vector<VertexRef>>(
              graph.outputs, [](auto &out) { return VertexRef(out); })) {}
};

class DfsVertRange : public VertRange<Vertex, DfsIter<Vertex>> {
public:
    DfsVertRange(const Graph &graph)
        : VertRange(Transform<std::vector<VertexRef>>(
              graph.inputs, [](auto &in) { return VertexRef(in); })) {}
};

/// Vertex visitor template that performs dynamic dispatch of visiting methods
template <class Ret, class... Args>
class VertexVisitor {
public:
    virtual Ret Visit(const VertexRef &vert, Args... args) {
        if (Contains(memo, vert)) return memo[vert];
        Ret ret;
        switch (vert->Kind()) {
            case VertexKind::INPUT:
                ret =
                    VisitInput(Cast<Input>(vert), std::forward<Args>(args)...);
                break;
            case VertexKind::OUTPUT:
                ret = VisitOutput(Cast<Output>(vert),
                                  std::forward<Args>(args)...);
                break;
            case VertexKind::OP:
                ret = VisitOp(Cast<Op>(vert), std::forward<Args>(args)...);
                break;
            default:
                LOG(FATAL) << "Unreachable.";
        }
        memo.insert({vert, ret});
        return ret;
    }

    virtual Ret VisitInput(const InputRef &input, Args... args) = 0;
    virtual Ret VisitOutput(const OutputRef &output, Args... args) = 0;
    virtual Ret VisitOp(const OpRef &op, Args... args) = 0;

protected:
    std::unordered_map<VertexRef, Ret> memo;
};

class VertexCloner : public VertexVisitor<VertexRef> {
public:
    VertexRef VisitInput(const InputRef &input) override;
    VertexRef VisitOutput(const OutputRef &output) override;
    VertexRef VisitOp(const OpRef &op) override;

    virtual ValueRef VisitValue(const ValueRef &value);

protected:
    std::unordered_map<ValueRef, ValueRef> valueMap;
};

}  // namespace hos