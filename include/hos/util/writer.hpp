#pragma once

#include <iostream>
#include <string>

class CodeWriter {
public:
    CodeWriter(std::ostream &ofs) : ofs(ofs) {}

    void Write(const std::string &str) { ofs << str; }

    void WriteLn(const std::string &str) {
        Write("\n");
        for (auto i = 0u; i < indCnt; i++) Write(INDENT_STR);
        Write(str);
    }

    class Indentation {
    public:
        Indentation(uint32_t &cnt) : cnt(cnt) { cnt++; }
        ~Indentation() { cnt--; }

    private:
        uint32_t &cnt;
    };

    Indentation Indent() { return Indentation(indCnt); }

private:
    static constexpr auto INDENT_STR = "    ";

    std::ostream &ofs;
    uint32_t indCnt = 0;
};
