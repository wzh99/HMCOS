#pragma once

#include <hos/core/graph.hpp>

namespace hos {

using OpSeq = std::vector<OpRef>;

/// Produce reverse post-order sequence of a computation graph
OpSeq ReversePostOrder(const Graph &graph);

/// Brute-force search the graph to find a topological order with optimal
/// metric.
/// The metric should also works for a continuous subsequence of topological
/// order. When a better sequence is found, the callback function is called.
void BruteForceSearch(const Graph &graph,
                      std::function<uint64_t(const OpSeq &)> metric,
                      std::function<void(const OpSeq &, uint64_t)> callback);

}  // namespace hos