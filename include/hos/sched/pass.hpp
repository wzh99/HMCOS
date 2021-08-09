#pragma once

#include <hos/core/hier.hpp>

namespace hos {

/// Passes that perform transformations on hierarchical graphs to enable
/// memory-aware scheduling

/// Join continunous sequences to form a larger sequence
class JoinSequencePass : public HierGraphPass {
public:
    void Run(HierGraph &graph) override;
};

class MakeGroupPass : public HierGraphPass {
public:
    void Run(HierGraph &graph) override;

    static bool intrusion;
    
    static std::function<bool(const SequenceRef &)> isCellOut;
};

}  // namespace hos