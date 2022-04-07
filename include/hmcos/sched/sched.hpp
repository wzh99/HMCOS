#pragma once

#include <hmcos/core/hier.hpp>
#include <random>

namespace hmcos {

void PlotSchedule(const std::vector<OpRef> &sched, const Graph &graph,
                  const std::string &dir, const std::string &name,
                  const std::string &format = "pdf");

/// Randomly sample a schedule of the computation graph
std::vector<OpRef> RandomSample(const Graph &graph, std::mt19937 &rng);

/// Produce reverse post-order sequence of a computation graph
std::vector<OpRef> ReversePostOrder(const Graph &graph);

/// Use iterative hierarchical scheduling algorithm of HMCOS
std::vector<OpRef> HierarchicalSchedule(const Graph &graph);

/// Serenity-style scheduling for networks with sequentially-connected cells
std::vector<OpRef> SerenitySchedule(const Graph &graph, bool joinOps,
                                    bool trySimple, size_t nSamples);

}  // namespace hmcos