#include "gguf.h"
#include "test_utils.h"
#include "tokenizer.h"

#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

// Tokenizer tests. We build a Tokenizer from a tiny tokenizer-only GGUF fixture
// (tests/gguf/tokenizer.gguf, made by tools/gen_tokenizer_fixture.py from the
// real TinyLlama checkpoint -- a few hundred KB, no tensors) and assert that
// encode() reproduces the genuine `sentencepiece` reference ids EXACTLY.
//
// The reference ids below were produced by tools/check_tokenizer.py against
// models/tok/tokenizer.model (sp.encode(s), no BOS). If encode() ever disagrees
// with sentencepiece, these literals are the ground truth -- the bug is ours.
//
// The fixture is gitignored (regenerate with `python3 tools/gen_tokenizer_fixture.py`),
// so we SKIP rather than fail the suite when it is absent. MAGI_GGUF_DIR is
// injected by CMake (CWD-independent).

namespace {
namespace fs = std::filesystem;

const fs::path GGUF_DIR = MAGI_GGUF_DIR;

// One fixture: the input text, sentencepiece's ids (no BOS), and whether the
// string round-trips losslessly (ASCII strings do; the unicode one exercises
// byte-fallback and survives a round-trip too, but we only require it for the
// ASCII set to keep the contract crisp).
struct Case {
    std::string      text;
    std::vector<int> ref_ids;  // from tools/check_tokenizer.py (sentencepiece)
    bool             check_roundtrip;
};

// Compare an int vector to expected and emit a [PASS]/[FAIL] line (reusing the
// vector check() from test_utils.h).
void check_ids(TestState& s, const std::string& name, const std::vector<int>& got,
               const std::vector<int>& expected) {
    check(s, name, got, expected);
}

}  // namespace

void run_tokenizer_tests(TestState& s) {
    const fs::path fixture = GGUF_DIR / "tokenizer.gguf";
    if (!fs::exists(fixture)) {
        std::cout << "  [SKIP] tokenizer tests -- run: python3 tools/gen_tokenizer_fixture.py\n";
        return;
    }

    gguf::GGUF g   = gguf::load_gguf(fixture.string());
    Tokenizer  tok = Tokenizer::from_gguf(g);

    // Special ids from the fixture (Llama convention).
    check_ids(s, "tokenizer: special ids (bos,eos,unk)",
              { tok.bos_id(), tok.eos_id(), tok.unk_id() }, { 1, 2, 0 });

    // Reference ids are sentencepiece's, baked from tools/check_tokenizer.py.
    const std::vector<Case> cases = {
        { "The capital of France is", { 450, 7483, 310, 3444, 338 }, true },
        { "Hello, world!", { 15043, 29892, 3186, 29991 }, true },
        { "def foo(x):\n    return x*2",
          { 822, 7953, 29898, 29916, 1125, 13, 1678, 736, 921, 29930, 29906 },
          true },
        { "TinyLlama is a 1.1B model.",
          { 323, 4901, 29931, 29880, 3304, 338, 263, 29871, 29896, 29889, 29896, 29933, 1904,
            29889 },
          true },
        // Unicode: 'café' (multibyte token), an emoji that has no token and so
        // hits byte-fallback (<0xE2><0x98><0x95>), and CJK tokens that ARE in vocab.
        { "café \xe2\x98\x95 \xe6\x97\xa5\xe6\x9c\xac\xe8\xaa\x9e",
          { 274, 28059, 29871, 229, 155, 152, 29871, 30325, 30346, 30968 },
          true },
    };

    for (const auto& c : cases) {
        // encode (no BOS) must match sentencepiece exactly.
        const std::vector<int> got = tok.encode(c.text, /*add_bos=*/false);
        check_ids(s, "tokenizer: encode '" + c.text + "' == sentencepiece", got, c.ref_ids);

        // round-trip: decode(encode(s)) == s.
        if (c.check_roundtrip) {
            const std::string back = tok.decode(got);
            check_ids(s, "tokenizer: round-trip '" + c.text + "'",
                      std::vector<int>{ back == c.text ? 1 : 0 }, std::vector<int>{ 1 });
            if (back != c.text) {
                std::cout << "    decode gave: '" << back << "'\n";
            }
        }
    }

    // add_bos prepends bos_id() and otherwise matches the no-BOS encoding.
    {
        const std::vector<int> no_bos   = tok.encode("Hello, world!", false);
        const std::vector<int> with_bos = tok.encode("Hello, world!", true);
        std::vector<int>       expected = { tok.bos_id() };
        expected.insert(expected.end(), no_bos.begin(), no_bos.end());
        check_ids(s, "tokenizer: add_bos prepends bos", with_bos, expected);
    }
}
