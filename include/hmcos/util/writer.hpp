#pragma once

#include <iostream>
#include <string>

namespace hmcos {

class CodeWriter {
public:
    CodeWriter(std::ostream &ofs) : ofs(ofs) {}

    void WriteLn(const std::string &str) {
        for (auto i = 0u; i < indCnt; i++) ofs << INDENT_STR;
        ofs << str << '\n';
    }

    void IncIndent() { indCnt++; }
    void DecIndent() { indCnt--; }

    class Indentation {
    public:
        Indentation(CodeWriter &writer) : writer(writer) { writer.IncIndent(); }
        ~Indentation() { writer.DecIndent(); }

    private:
        CodeWriter &writer;
    };

    Indentation Indent() { return Indentation(*this); }

private:
    static constexpr auto INDENT_STR = "    ";

    std::ostream &ofs;
    uint32_t indCnt = 0;
};

}