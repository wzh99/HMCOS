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
    explicit MakeGroupPass(bool intrude) : intrude(intrude) {}
    
    void Run(HierGraph &graph) override;

private:
    /// This method defines how output of a cell is identified
    static bool isCellOut(const SequenceRef &seq) {
        return seq->ops.front()->type == "Concat";
    }

    bool intrude;
};

}  // namespace hos