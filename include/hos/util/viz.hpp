#pragma once

#include <glog/logging.h>

#include <map>
#include <string>
#include <unordered_map>

/// API for defining a directed graph in Graphviz DOT language.
/// This class is just for visualization of computation graph in the project,
/// and may not cover further use cases.
template <class NodeType>
class DotCreator {
public:
    DotCreator(const std::string &name) : name(name) {}
    template <class F>
    void AddNodes(const std::vector<NodeType> &nodes, F getName);
    void AddEdge(const NodeType &tail, const NodeType &head);
    void Render(const std::string &dir, const std::string &format) const;

private:
    std::string name;
    std::unordered_map<NodeType, uint32_t> nodeIds;
    std::vector<std::string> nodeNames;
    std::vector<std::pair<uint32_t, uint32_t> > edges;
};