// magi_run -- standalone GGUF inference entrypoint. - by opus
//
// Loads a real GGUF checkpoint, builds a Model from it, and runs a single
// forward pass (and optionally greedy generation) over a fixed prompt. This is
// the program we use both as the live-demo entrypoint and as the C++ half of
// the numerical-correctness gate: with --dump-logits it writes the last-row
// logits to disk in the same raw little-endian float32 format the NumPy
// reference (tools/gen_real_reference.py) emits, so the two can be diffed
// element-for-element.
//
// Usage:
//   magi_run <model.gguf> [--ids 1,22172,...] [--prompt "text"]
//            [--dump-logits <path>] [--gen N]
//
//   --ids          comma-separated token ids (default: the fixed gate prompt)
//   --prompt TEXT  tokenize TEXT (SPM, with BOS) and use those ids instead;
//                  prints the segmentation. Overrides --ids.
//   --dump-logits  write last-row logits (vocab floats, LE f32) to <path>
//   --gen N        after the forward, greedily decode N tokens; with --prompt,
//                  streams the decoded text continuation live and stops at EOS

#include "gguf.h"
#include "gguf_model.h"
#include "model.h"
#include "model_config.h"
#include "ops.h"
#include "tensor.h"
#include "tokenizer.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// The fixed gate prompt. Must stay identical to IDS in tools/gen_real_reference.py
// or the cross-implementation comparison is meaningless.
const std::vector<int> DEFAULT_IDS = { 1, 22172, 29892, 590, 1024 };

// Parse "1,22172,29892" into {1, 22172, 29892}. Throws on a malformed entry.
std::vector<int> parse_ids(const std::string& s) {
    std::vector<int> ids;
    std::size_t      start = 0;
    while (start <= s.size()) {
        std::size_t comma = s.find(',', start);
        std::string tok =
            s.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
        if (!tok.empty()) {
            ids.push_back(std::stoi(tok));
        }
        if (comma == std::string::npos) {
            break;
        }
        start = comma + 1;
    }
    if (ids.empty()) {
        throw std::runtime_error("--ids: no token ids parsed");
    }
    return ids;
}

// Drop trailing whitespace. SentencePiece attaches a space to the *following*
// word, so a trailing space has no word to attach to and tokenizes to a lone
// meta-space token (e.g. 29871) -- a dangling boundary that derails greedy
// decoding (the model then predicts a newline). It's never what you want.
std::string rstrip(const std::string& s) {
    const std::size_t end = s.find_last_not_of(" \t\r\n");
    return end == std::string::npos ? std::string() : s.substr(0, end + 1);
}

// Indices of the top-k largest values in `row`, descending by value.
std::vector<int> topk(const float* row, int n, int k) {
    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    k = std::min(k, n);
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                      [&](int a, int b) { return row[a] > row[b]; });
    idx.resize(k);
    return idx;
}

// Write `n` floats to `path` as raw little-endian float32. We're on a
// little-endian host (arm64/x86-64), so a straight byte dump is already LE.
void dump_floats(const std::string& path, const float* data, int n) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("could not open --dump-logits path: " + path);
    }
    out.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(n) * sizeof(float));
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: " << argv[0]
                  << " <model.gguf> [--ids 1,22172,...] [--prompt \"text\"] "
                     "[--dump-logits <path>] [--gen N]\n";
        return 1;
    }

    const std::string model_path = argv[1];
    std::vector<int>  ids        = DEFAULT_IDS;
    std::string       dump_path;
    std::string       prompt;
    bool              have_prompt = false;
    int               gen_n       = 0;

    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--ids" && i + 1 < argc) {
            ids = parse_ids(argv[++i]);
        } else if (arg == "--prompt" && i + 1 < argc) {
            prompt      = argv[++i];
            have_prompt = true;
        } else if (arg == "--dump-logits" && i + 1 < argc) {
            dump_path = argv[++i];
        } else if (arg == "--gen" && i + 1 < argc) {
            gen_n = std::stoi(argv[++i]);
        } else {
            std::cerr << "unknown or incomplete argument: " << arg << "\n";
            return 1;
        }
    }

    try {
        // --- Load + build. Time the load separately: for an F32 multi-GB file the
        // mmap is cheap but the first forward pays the page-in cost, so isolating
        // load here keeps the forward timing honest. We snapshot the config off `g`
        // BEFORE moving it into the Model (build_model_from_gguf consumes `g`). ---
        const auto t_load0 = std::chrono::steady_clock::now();
        gguf::GGUF        g   = gguf::load_gguf(model_path);
        const ModelConfig cfg = config_from_gguf(g);

        // Build the tokenizer off the still-live GGUF (it copies the tokenizer.*
        // metadata it needs) BEFORE build_model_from_gguf consumes `g` by move.
        // Only needed for --prompt / decoding generated text.
        std::optional<Tokenizer> tok;
        if (have_prompt) {
            tok = Tokenizer::from_gguf(g);
        }

        Model             m       = build_model_from_gguf(std::move(g));
        const auto        t_load1 = std::chrono::steady_clock::now();
        const double load_s =
            std::chrono::duration<double>(t_load1 - t_load0).count();

        std::cout << "model: " << model_path << "\n";
        std::cout << "config: n_layers=" << cfg.n_layers << " hidden=" << cfg.hidden_size
                  << " intermediate=" << cfg.intermediate_size << " n_heads=" << cfg.n_heads
                  << " n_kv_heads=" << cfg.n_kv_heads << " head_dim=" << cfg.head_dim
                  << " max_seq=" << cfg.max_seq_len << " rope_base=" << cfg.rope_base
                  << " rms_eps=" << cfg.rms_eps << " vocab=" << cfg.vocab_size << "\n";
        std::cout << "load + page-in (build): " << load_s << " s\n";

        // --- --prompt: tokenize text -> ids (with BOS) and show the segmentation
        // so we can eyeball the SPM split. Overrides any --ids. ---
        if (have_prompt) {
            const std::string trimmed = rstrip(prompt);
            if (trimmed.size() != prompt.size()) {
                std::cout << "note: stripped trailing whitespace from --prompt "
                             "(it would tokenize to a dangling space token)\n";
                prompt = trimmed;
            }
            ids = tok->encode(prompt, /*add_bos=*/true);
            std::cout << "prompt: " << prompt << "\n";
            std::cout << "encode -> " << ids.size() << " tokens:\n";
            for (int id : ids) {
                std::cout << "  " << id << "  '" << tok->decode_token(id) << "'\n";
            }
        }

        std::cout << "ids (T=" << ids.size() << "): [";
        for (std::size_t i = 0; i < ids.size(); ++i) {
            std::cout << ids[i] << (i + 1 < ids.size() ? ", " : "");
        }
        std::cout << "]\n";

        // --- Single forward pass, timed. ---
        const int T = static_cast<int>(ids.size());
        Tensor    logits({ T, cfg.vocab_size });

        std::cout << "\nrunning forward over " << T
                  << " tokens (single-threaded, no KV cache)..." << std::flush;
        const auto t_fwd0 = std::chrono::steady_clock::now();
        m.forward(ids, logits);
        const auto   t_fwd1 = std::chrono::steady_clock::now();
        const double fwd_s  = std::chrono::duration<double>(t_fwd1 - t_fwd0).count();

        // Last row of logits: shape (vocab,) at offset (T-1)*vocab.
        const float* last = logits.data_ptr() + static_cast<std::size_t>(T - 1) * cfg.vocab_size;

        const int argmax_id = ops::argmax(logits);  // argmax of the LAST row
        std::cout << "\nforward: " << fwd_s << " s   (~" << (fwd_s > 0 ? T / fwd_s : 0.0)
                  << " tok/s for T=" << T << ")\n";
        std::cout << "argmax token id = " << argmax_id << "   logit = " << last[argmax_id] << "\n";

        // With a tokenizer in hand, decode the predicted next token to text -- the
        // end-to-end "text in -> sensible next word" signal (e.g. "France is" -> " Paris").
        if (have_prompt) {
            std::cout << "argmax next token (decoded) = '" << tok->decode_token(argmax_id) << "'\n";
        }

        std::cout << "top-5 (id, logit):\n";
        for (int id : topk(last, cfg.vocab_size, 5)) {
            std::cout << "  " << id << "  " << last[id] << "\n";
        }

        if (!dump_path.empty()) {
            dump_floats(dump_path, last, cfg.vocab_size);
            std::cout << "dumped last-row logits (" << cfg.vocab_size << " floats) -> " << dump_path
                      << "\n";
        }

        // --- Optional greedy generation. ---
        if (gen_n > 0) {
            const auto t_gen0 = std::chrono::steady_clock::now();

            if (have_prompt) {
                // Stream the continuation live. We reuse the forward already run above
                // (its argmax is the first generated token), then after each step decode
                // the WHOLE sequence and print only the newly-added text. Decoding the
                // full string each step -- rather than token-by-token -- is what makes
                // SentencePiece's meta-space and multi-byte / byte-fallback pieces render
                // correctly (a single character can span several byte tokens). There is no
                // KV cache yet, so every step re-runs the whole prefix: the streaming is
                // purely so a slow run doesn't look frozen.
                std::cout << "\n=== generation (greedy) ===\n";

                std::vector<int> out     = ids;
                std::string      shown   = tok->decode(out);  // the prompt text so far
                std::cout << shown << std::flush;
                std::size_t      printed = shown.size();

                int  next      = argmax_id;  // first token: already computed above
                int  generated = 0;
                bool hit_eos   = false;
                while (generated < gen_n) {
                    out.push_back(next);
                    ++generated;

                    // Re-decode the full sequence and emit only the new suffix.
                    const std::string full = tok->decode(out);
                    if (full.size() > printed) {
                        std::cout << full.substr(printed) << std::flush;
                        printed = full.size();
                    }

                    if (next == tok->eos_id()) {
                        hit_eos = true;
                        break;
                    }
                    if (generated == gen_n) {
                        break;  // enough tokens -- no need to forward again
                    }

                    // No KV cache: re-forward the whole grown sequence for the next token.
                    Tensor step_logits({ static_cast<int>(out.size()), cfg.vocab_size });
                    m.forward(out, step_logits);
                    next = ops::argmax(step_logits);
                }

                const double gen_s =
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - t_gen0).count();
                std::cout << (hit_eos ? "  [eos]\n" : "\n");
                std::cout << "[" << generated << " tokens in " << gen_s << " s, ~"
                          << (gen_s > 0 ? generated / gen_s : 0.0) << " tok/s]\n";
            } else {
                // No prompt/tokenizer (raw --ids path): greedy by ids only.
                const std::vector<int> out = m.generate(ids, gen_n);
                const double           gen_s =
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - t_gen0).count();
                std::cout << "\ngenerated " << (out.size() - ids.size()) << " tokens in " << gen_s
                          << " s\nfull id sequence: [";
                for (std::size_t i = 0; i < out.size(); ++i) {
                    std::cout << out[i] << (i + 1 < out.size() ? ", " : "");
                }
                std::cout << "]\n";
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
