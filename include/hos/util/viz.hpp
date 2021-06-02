#pragma once

#include <glog/logging.h>

#include <filesystem>
#include <fstream>
#include <hos/util/writer.hpp>
#include <unordered_map>

namespace hos {

#if defined(WIN32)
#define DEFAULT_FONT "Segoe UI"
#elif defined(__APPLE__)
#define DEFAULT_FONT ".AppleSystemUIFont"
#endif

/// API for defining a directed graph in Graphviz DOT language.
/// This class is just for visualization of computation graph in the project,
/// and may not cover further use cases.
template <class NodeType>
class DotCreator {
public:
    DotCreator(const std::string &name) : name(name) {}
    void AddNode(const NodeType &node, const std::string &label);
    void AddEdge(const NodeType &tail, const NodeType &head);
    void Render(const std::string &dir, const std::string &format) const;

private:
    std::string name;
    std::unordered_map<NodeType, uint32_t> nodeIds;
    std::vector<std::string> nodeLabels;
    std::vector<std::pair<uint32_t, uint32_t> > edges;
};

template <class NodeType>
void DotCreator<NodeType>::AddNode(const NodeType &node,
                                   const std::string &label) {
    if (Contains(nodeIds, node)) return;
    auto id = uint32_t(nodeIds.size());
    nodeIds.insert({node, id});
    nodeLabels.push_back(label);
}

template <class NodeType>
inline void DotCreator<NodeType>::AddEdge(const NodeType &tail,
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
    writer.Write("digraph " + name + " { ");
    {
        // Write node attribute
        auto indent = writer.Indent();
        auto nodeAttr =
            fmt::format("node [fontname={} shape=box style=rounded]",
                        FmtStr(DEFAULT_FONT, '"'));
        writer.WriteLn(nodeAttr);

        // Write nodes
        for (auto i = 0u; i < nodeLabels.size(); i++)
            writer.WriteLn(fmt::format("{} [label=\"{}\"]", i,
                                       FmtStr(nodeLabels[i], '"')));

        // Write edges
        for (auto &[tail, head] : edges)
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