#include "test_utils.h"
#include "mmap.h"
#include "tensor.h"

#include <filesystem>
#include <fstream>
#include <memory>
#include <system_error>
#include <vector>

// Writes `vals` as raw little-endian float32 into a temp file and returns its
// path. This mirrors tools/gen_dummy_weights.py, but inline so the test owns
// its fixture: no committed binary blob, no dependency on the current working
// directory. NOTE: writing float bytes directly assumes a little-endian host
// (true on x86 and Apple Silicon); a real loader would byte-swap if needed.
static std::string write_dummy_file(const std::vector<float>& vals) {
    namespace fs = std::filesystem;
    fs::path p = fs::temp_directory_path() / "magi_dummy_weights.bin";
    std::ofstream out(p, std::ios::binary | std::ios::trunc);
    out.write(reinterpret_cast<const char*>(vals.data()),
              static_cast<std::streamsize>(vals.size() * sizeof(float)));
    out.close();
    return p.string();
}

// Local float-array comparison. test_utils.h's check() compares against a
// Tensor or an int vector; here we sometimes have a bare float* (the mapped
// region) with no Tensor yet, so this fills the gap in the same PASS/FAIL style.
static void check_floats(TestState& s, const std::string& name,
                         const float* got, const std::vector<float>& expected,
                         float tol = 1e-6f) {
    bool ok = true;
    for (size_t i = 0; ok && i < expected.size(); ++i)
        if (std::fabs(got[i] - expected[i]) > tol) ok = false;
    std::cout << (ok ? "  [PASS] " : "  [FAIL] ") << name << "\n";
    if (ok) s.passed++; else s.failed++;
}

void run_mmap_tests(TestState& s) {
    const std::vector<float> vals = {1, 2, 3, 4, 5, 6};
    const std::string path = write_dummy_file(vals);

    // --- 4a: MappedFile round-trip (no dependency on the Tensor refactor) ---

    // Error path: mapping a file that does not exist must throw. This exercises
    // the open() failure branch in mmap.cpp.
    bool threw = false;
    try {
        MappedFile bad("/this/path/should/not/exist/magi.bin");
    } catch (const std::system_error&) {
        threw = true;
    }
    check(s, "missing file throws", std::vector<int>{threw ? 1 : 0},
          std::vector<int>{1});

    MappedFile mf(path);

    // Size is in bytes: 6 floats * 4 bytes = 24.
    check(s, "mmap size in bytes",
          std::vector<int>{static_cast<int>(mf.size())}, std::vector<int>{24});

    // Reinterpret the mapped bytes as float32 and confirm they match what we
    // wrote. data() is page-aligned (offset 0), so the cast is well-defined.
    const float* fp = reinterpret_cast<const float*>(mf.data());
    check_floats(s, "mmap bytes decode to expected floats", fp, vals);

    // --- 4b: mmap-backed Tensor view ---
    // REQUIRES your step-2 view constructor, signature:
    //   Tensor(std::vector<int> shape, const float* data,
    //          std::shared_ptr<void> owner);
    // Uncomment once that exists. The mapping is held by `mapping`; the Tensor's
    // owner_ keeps it alive, so munmap fires only when the last ref is gone.
    //
    // auto mapping = std::make_shared<MappedFile>(path);
    // const float* base = reinterpret_cast<const float*>(mapping->data());
    // Tensor view({2, 3}, base, mapping);
    // check(s, "mmap-backed tensor view", view, vals, {2, 3});

    std::error_code ec;
    std::filesystem::remove(path, ec);  // best-effort cleanup
}
