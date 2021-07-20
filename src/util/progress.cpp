#include <fmt/chrono.h>

#include <hos/util/progress.hpp>

namespace hos {

static constexpr auto BAR_LENGTH = 50u;

using namespace std::chrono;

void PrintProgress(size_t index, size_t size, system_clock::time_point start) {
    // Print progress bar
    printf("\r");
    auto nDone = size_t(float(index) / size * BAR_LENGTH);
    for (auto i = 0u; i < nDone; i++) printf("*");
    for (auto i = nDone; i < BAR_LENGTH; i++) printf("-");

    // Print time estimate
    if (index == 0)
        printf("00:00:00|--:--:--");
    else {
        auto dur = system_clock::now() - start;
        auto rem = dur * float(size - index) / index;
        fmt::print("{:%H:%M:%S}|{:%H:%M:%S}", duration_cast<seconds>(dur),
                   duration_cast<seconds>(rem));
    }

    // Flush standard output
    fflush(stdout);
}

}  // namespace hos