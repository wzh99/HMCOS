#include <hos/util/fmt.hpp>
#include <hos/util/viz.hpp>

namespace hos {

static const auto PYTHON_PREAMBLE =
    "import matplotlib as mpl\n"
    "import matplotlib.pyplot as plt\n"
    "import matplotlib.ticker as mtick\n\n";

static std::unordered_map<std::string, std::string> rcParams{
    {"figure.figsize", "(10, 6)"},
    {"figure.dpi", "150"},
    {"font.sans-serif", FmtStr(DEFAULT_FONT)},
};

static const char *colors[] = {
    "aqua",        "darkturquoise",  "cadetblue",      "powderblue",
    "deepskyblue", "skyblue",        "lightskyblue",   "steelblue",
    "dodgerblue",  "lightsteelblue", "cornflowerblue", "royalblue",
};

static constexpr auto NUM_COLORS = sizeof(colors) / sizeof(colors[0]);

void RectPlot::AddRect(float coordX, float coordY, float width, float height) {
    rects.push_back({{coordX, coordY}, width, height});
    xMin = std::min(xMin, coordX);
    yMin = std::min(yMin, coordY);
    xMax = std::max(xMax, coordX + width);
    yMax = std::max(yMax, coordY + height);
}

void hos::RectPlot::Render(const std::string &dir,
                           const std::string &format) const {
    // Create Python file
    using namespace std::filesystem;
    auto pyPath = (path(dir) / path(name + ".py")).string();
    std::ofstream ofs(pyPath);
    if (!ofs.is_open()) {
        LOG(ERROR) << fmt::format("Cannot create Python source file '{}'.",
                                  pyPath);
        return;
    }

    // Emit Python code for visualization
    CodeWriter writer(ofs);
    // Preamble
    writer.Write(PYTHON_PREAMBLE);
    // Parameters
    for (auto &[key, val] : rcParams)
        writer.WriteLn(fmt::format("mpl.rcParams[{}] = {}", FmtStr(key), val));
    writer.WriteLn("ax = plt.gca()");
    // Limits
    writer.WriteLn(fmt::format("plt.xlim({}, {})", xMin, xMax));
    writer.WriteLn(fmt::format("plt.ylim({}, {})", yMin, yMax));
    // Plot rectangles
    auto colorIdx = 0u;
    for (auto &rect : rects) {
        writer.WriteLn(
            fmt::format("ax.add_patch(plt.Rectangle(({}, {}), {}, {}, "
                        "facecolor={}))",
                        rect.coord.first, rect.coord.second, rect.width,
                        rect.height, FmtStr(colors[colorIdx])));
        colorIdx = (colorIdx + 1) % NUM_COLORS;
    }
    // Save figure
    auto figPath = path(dir) / path(fmt::format("{}.{}", name, format));
    writer.WriteLn(fmt::format("plt.savefig({})", FmtStr(figPath.string())));
    ofs.close();

    // Execute Python code
    auto cmd = fmt::format("python {}", pyPath);
    auto ret = system(cmd.c_str());
    if (ret != 0)
        LOG(ERROR) << fmt::format("Cannot run Python script '{}'.", pyPath);
}

}  // namespace hos