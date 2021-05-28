#pragma once

#include <hos/core/graph.hpp>

namespace hos {

using OpSeq = std::vector<OpRef>;

OpSeq ReversePostOrder(const Graph &graph);

void BruteForceSearch(const Graph &graph, std::function<double(const OpSeq &)> metric,
                      std::function<void(const OpSeq &, double)> callback);

}  // namespace hos