#pragma once

#include <hos/core/graph.hpp>

namespace hos {

using OpSeq = std::vector<OpRef>;

OpSeq ReversePostOrder(const Graph &graph);

void BruteForceSearch(const Graph &graph, std::function<uint64_t(const OpSeq &)> metric,
                      std::function<void(const OpSeq &, uint64_t)> callback);

}  // namespace hos