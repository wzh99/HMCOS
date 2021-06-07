#pragma once

#include <glog/logging.h>

#include <filesystem>
#include <fstream>
#include <hos/util/writer.hpp>
#include <hos/util/fmt.hpp>
#include <unordered_map>

namespace hos {

#if defined(WIN32)
#define DEFAULT_FONT "Segoe UI"
#elif defined(__APPLE__)
#define DEFAULT_FONT "Helvetica"
#endif

/// API for defining a directed graph in Graphviz DOT language.
/// This class is just for visualization of computation graph, as well as
/// hierarchical graph in the project, and may not cover further use cases.
template <class NodeType>
class DotCreator {
public:
    DotCreator(const std::string &name) : name(name) {}
    void Node(const NodeType &node, const std::string &label);
    void Edge(const NodeType &tail, const NodeType &head);
    void Render(const std::string &dir, const std::string &format) const;

    enum class NodeKind {
        NORMAL,
        CLUSTER,
    };

    struct NodeData {
        /// Kind of this node
        NodeKind kind;
        /// ID of this node. Use separate counter for normal nodes and clusters.
        uint32_t id;
        /// Node at this level, valid for clusters
        std::unique_ptr<std::vector<NodeData>> nodes;
    };

    class Context {
    public:
        void Node(const NodeType &node, const std::string &label);
        Context Cluster();

    private:
        Context(DotCreator &creator, std::vector<NodeData> &nodes)
            : creator(creator), nodes(nodes) {}

        DotCreator &creator;
        /// Node data reference of this cluster
        std::vector<NodeData> &nodes;

        friend class DotCreator;
    };

    /// Top level context of this graph
    Context Top();

private:
    uint32_t registerNode(const NodeType &node, const std::string &label);
    uint32_t registerCluster();

    void writeData(CodeWriter &writer, const NodeData &data) const;

    static constexpr auto INVALID_ID = UINT32_MAX;

    /// Name of this plot
    std::string name;
    /// Map normal nodes to its ids
    std::unordered_map<NodeType, uint32_t> nodeIds;
    /// String labels of normal nodes
    std::vector<std::string> nodeLabels;
    /// Top level nodes
    std::vector<NodeData> top;
    /// Number of clusters in the whole graph
    uint32_t nClusters = 0;
    /// Edges connecting normal nodes
    std::vector<std::pair<uint32_t, uint32_t>> edges;

    friend class Context;
};

template <class NodeType>
inline void DotCreator<NodeType>::Context::Node(const NodeType &node,
                                                const std::string &label) {
    auto id = creator.registerNode(node, label);
    if (id != INVALID_ID) nodes.push_back({NodeKind::NORMAL, id, nullptr});
}

template <class NodeType>
inline typename DotCreator<NodeType>::Context
DotCreator<NodeType>::Context::Cluster() {
    auto id = creator.registerCluster();
    nodes.push_back(
        {NodeKind::CLUSTER, id, std::make_unique<std::vector<NodeData>>()});
    return Context(creator, *nodes.back().nodes);
}

template <class NodeType>
inline typename DotCreator<NodeType>::Context DotCreator<NodeType>::Top() {
    return Context(*this, top);
}

template <class NodeType>
inline uint32_t DotCreator<NodeType>::registerNode(const NodeType &node,
                                                   const std::string &label) {
    if (Contains(nodeIds, node)) return INVALID_ID;
    auto id = uint32_t(nodeIds.size());
    nodeIds.insert({node, id});
    nodeLabels.push_back(label);
    return id;
}

template <class NodeType>
inline uint32_t DotCreator<NodeType>::registerCluster() {
    return nClusters++;
}

template <class NodeType>
inline void DotCreator<NodeType>::Node(const NodeType &node,
                                       const std::string &label) {
    auto id = registerNode(node, label);
    if (id != INVALID_ID) top.push_back({NodeKind::NORMAL, id, nullptr});
}

template <class NodeType>
inline void DotCreator<NodeType>::Edge(const NodeType &tail,
                                       const NodeType &head) {
    if (!Contains(nodeIds, tail)) {
        LOG(ERROR) << "Tail node has not been added.";
        return;
    }
    if (!Contains(nodeIds, head)) {
        LOG(ERROR) << "Head node has not been added.";
        return;
    }
    edges.emplace_back(nodeIds[tail], nodeIds[head]);
}

template <class NodeType>
void DotCreator<NodeType>::writeData(
    CodeWriter &writer, const DotCreator<NodeType>::NodeData &data) const {
    if (data.kind == NodeKind::NORMAL)
        writer.WriteLn(fmt::format("{} [label={}]", data.id,
                                   FmtStr(nodeLabels[data.id], '"')));
    else {
        writer.WriteLn(fmt::format("subgraph cluster{}", data.id) + " {");
        {
            auto indent = writer.Indent();
            for (auto &node : *data.nodes) writeData(writer, node);
        }
        writer.WriteLn("}");
    }
}

template <class NodeType>
void DotCreator<NodeType>::Render(const std::string &dir,
                                  const std::string &format) const {
    // Create DOT source file
    using namespace std::filesystem;
    auto srcPath = (path(dir) / path(name + ".gv")).string();
    std::ofstream ofs(srcPath);
    if (!ofs.is_open()) {
        LOG(ERROR) << fmt::format("Cannot create source file: {}", srcPath);
        return;
    }

    // Write code to file
    CodeWriter writer(ofs);
    writer.Write("digraph " + FmtStr(name, '"') + " {");
    {
        // Write node attribute
        auto indent = writer.Indent();
        auto nodeAttr =
            fmt::format("node [fontname={} shape=box style=rounded]",
                        FmtStr(DEFAULT_FONT, '"'));
        writer.WriteLn(nodeAttr);

        // Write nodes
        for (auto &data : top) writeData(writer, data);

        // Write edges
        for (auto [tail, head] : edges)
            writer.WriteLn(fmt::format("{} -> {}", tail, head));
    }
    writer.WriteLn("}");
    ofs.close();

    // Compile source with `dot` command
    auto cmd = fmt::format("dot -T{} -O {}", format, srcPath);
    auto ret = system(cmd.c_str());
    if (ret != 0)
        LOG(ERROR) << fmt::format("Cannot compile source file {}", srcPath);
}

struct Rect {
    /// Bottom-left coordinate (x, y) of the rectangle
    std::pair<float, float> coord;
    /// Width of the rectangle (X-axis)
    float width;
    /// Height of the rectangle (Y-axis)
    float height;
};

/// Plot rectangles
class RectPlot {
public:
    RectPlot(const std::string &name) : name(name) {}
    void AddRect(float coordX, float coordY, float width, float height);
    void Render(const std::string &dir, const std::string &format) const;

private:
    std::string name;
    std::vector<Rect> rects;
    float xMin = 0.f, xMax = 0.f, yMin = 0.f, yMax = 0.f;
};

}  // namespace hos