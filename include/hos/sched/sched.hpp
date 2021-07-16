#pragma once

#include <hos/core/hier.hpp>
#include <random>

namespace hos {

void PlotSchedule(const std::vector<OpRef> &sched, const Graph &graph,
                  const std::string &dir, const std::string &name,
                  const std::string &format = "pdf");

/// Randomly sample a schedule of the computation graph
std::vector<OpRef> RandomSample(const Graph &graph, std::mt19937 &rng);

/// Produce reverse post-order sequence of a computation graph
std::vector<OpRef> ReversePostOrder(const Graph &graph);

/// Schedule ops on a hierarchical graph
std::vector<OpRef> HierarchicalSchedule(const Graph &graph);

}  // namespace hos