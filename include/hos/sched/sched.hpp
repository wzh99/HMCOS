#pragma once

#include <hos/core/graph.hpp>

namespace hos {

using OpSeq = std::vector<OpRef>;

OpSeq ReversePostOrder(const Graph &graph);

}  // namespace hos