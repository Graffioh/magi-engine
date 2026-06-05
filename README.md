# magi-engine

<img width="700" height="458" alt="image" src="https://github.com/user-attachments/assets/815c6192-0d39-481b-a9c8-07e23d8d165e" />

CPU-only C++ inference engine for Tinyllama.

## Running inference

Runs real [TinyLlama-1.1B-Chat](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)
from a GGUF checkpoint.

**1. Get the model (one-time).** Downloads the Q8_0 GGUF and dequantizes it to a
plain F32 GGUF the loader can mmap (`models/tinyllama-1.1b-chat-f32.gguf`, ~4.4 GB):

```sh
pip install gguf huggingface_hub
python3 tools/make_f32_gguf.py
```

**2. Build the optimized runner.** The default `build/` is unoptimized; a 22-layer
F32 forward needs a release build:

```sh
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release
cmake --build build-release --target magi_run
```

**3. Run.** Give it a prompt and how many tokens to generate:

```sh
./build-release/magi_run models/tinyllama-1.1b-chat-f32.gguf \
    --prompt "The capital of France is" --gen 10
# -> The capital of France is Paris.
```

`--prompt` tokenizes the text (SPM, with BOS) and prints the segmentation; `--gen N`
greedily decodes N tokens. Also: `--ids 1,450,...` to feed raw token ids,
`--dump-logits <path>` to write the last-row logits.

Example run (note the trailing space in the prompt being auto-stripped):

```text
$ ./build-release/magi_run models/tinyllama-1.1b-chat-f32.gguf --prompt "The capital of Italy is " --gen 10
model: models/tinyllama-1.1b-chat-f32.gguf
config: n_layers=22 hidden=2048 intermediate=5632 n_heads=32 n_kv_heads=4 head_dim=64 max_seq=2048 rope_base=10000 rms_eps=1e-05 vocab=32000
load + page-in (build): 0.013511 s
note: stripped trailing whitespace from --prompt (it would tokenize to a dangling space token)
prompt: The capital of Italy is
encode -> 6 tokens:
  1  '<s>'
  450  ' The'
  7483  ' capital'
  310  ' of'
  12730  ' Italy'
  338  ' is'
ids (T=6): [1, 450, 7483, 310, 12730, 338]

running forward over 6 tokens (single-threaded, no KV cache)...
forward: 20.1751 s   (~0.297396 tok/s for T=6)
argmax token id = 9184   logit = 13.3954
argmax next token (decoded) = ' Rome'
top-5 (id, logit):
  9184  13.3954
  29973  10.5742
  903  10.3448
  5982  10.3344
  20308  10.2916

=== generation (greedy) ===
The capital of Italy is Rome.
```
