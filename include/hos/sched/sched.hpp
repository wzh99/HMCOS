#pragma once

#include <hos/core/hier.hpp>

namespace hos {

void PlotSchedule(const std::vector<OpRef> &sched, const Graph &graph,
                  const std::string &dir, const std::string &name,
                  const std::string &format = "pdf");

/// Produce reverse post-order sequence of a computation graph
std::vector<OpRef> ReversePostOrder(const Graph &graph);

/// Brute-force search the graph to find a topological order with optimal
/// metric.
/// The metric should also works for a continuous subsequence of topological
/// order. When a better sequence is found, the callback function is called.
void BruteForceSearch(
    const Graph &graph,
    std::function<uint64_t(const std::vector<OpRef> &)> metric,
    std::function<void(const std::vector<OpRef> &, uint64_t)> callback);

/// Schedule ops on a hierarchical graph
std::vector<OpRef> HierarchicalSchedule(const HierGraph &hier);

}  // namespace hos