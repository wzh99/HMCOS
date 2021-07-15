#include <hos/util/fmt.hpp>
#include <hos/util/viz.hpp>

namespace hos {

#if defined(WIN32)
#define PYTHON_CMD "python"
#elif defined(__APPLE__)
#define PYTHON_CMD "python3"
#endif

static const auto PYTHON_PREAMBLE =
    "import matplotlib as mpl\n"
    "import matplotlib.pyplot as plt\n\n";

static std::vector<std::pair<std::string, std::string>> rcParams{
    {"figure.figsize", "(8, 6)"},
    {"figure.dpi", "150"},
    {"font.sans-serif", FmtStr(DEFAULT_FONT)},
};

void PythonPlot::Render(const std::string &dir,
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
    writer.WriteLn(PYTHON_PREAMBLE);
    // Parameters
    for (auto &[key, val] : rcParams)
        writer.WriteLn(fmt::format("mpl.rcParams[{}] = {}", FmtStr(key), val));
    // Main code
    writeMain(writer);
    // Save figure
    auto figPath = path(dir) / path(fmt::format("{}.{}", name, format));
    writer.WriteLn(fmt::format("plt.savefig({})", FmtStr(figPath.string())));
    ofs.close();

    // Execute Python code
    auto cmd = fmt::format("{} {}", PYTHON_CMD, pyPath);
    auto ret = system(cmd.c_str());
    if (ret != 0)
        LOG(ERROR) << fmt::format("Cannot run Python script '{}'.", pyPath);
}

void RectPlot::AddRect(float coordX, float coordY, float width, float height,
                       const char *color) {
    rects.push_back({{coordX, coordY}, width, height, color});
    xMin = std::min(xMin, coordX);
    yMin = std::min(yMin, coordY);
    xMax = std::max(xMax, coordX + width);
    yMax = std::max(yMax, coordY + height);
}

void RectPlot::writeMain(CodeWriter &writer) const {
    // Limits
    writer.WriteLn("ax = plt.gca()");
    writer.WriteLn(fmt::format("plt.xlim({}, {})", xMin, xMax));
    writer.WriteLn(fmt::format("plt.ylim({}, {})", yMin, yMax));
    // Plot rectangles
    for (auto &rect : rects)
        writer.WriteLn(
            fmt::format("ax.add_patch(plt.Rectangle(({}, {}), {}, {}, "
                        "facecolor={}))",
                        rect.coord.first, rect.coord.second, rect.width,
                        rect.height, FmtStr(rect.color)));
}

void HistoPlot::writeMain(CodeWriter &writer) const {
    writer.WriteLn(FmtList(
        data, [](float f) { return fmt::format("{}", f); }, "a = [", "]"));
    writer.WriteLn("print(min(a), max(a), sum(a) / len(a))");
    writer.WriteLn("plt.hist(a, 50)");
}

}  // namespace hos