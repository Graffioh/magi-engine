#pragma once

#include "gguf.h"
#include "model.h"
#include "model_config.h"

// GGUF -> Model adapter. The loader in gguf.h hands us a parsed container
// (typed metadata kv block + named mmap tensor views). This turns that generic
// bag into a runnable Model: it reads the llama.* hyperparameters into a
// ModelConfig and wires each ggml-named weight into the right layer, moving the
// tensor views (not copying the data) so the Model keeps the mmap alive on its
// own after the GGUF is gone.
//
// Only the "llama" architecture is handled -- enough for the toy round-trip and
// any F32 Llama-family checkpoint our loader can read.

// Read the hyperparameters from the llama.* metadata into a ModelConfig.
// Throws (via the typed accessors) if a required key is missing.
ModelConfig config_from_gguf(const gguf::GGUF& g);

// Consume `g` and build a Model from it. Tensor views are MOVED out of
// g.tensors into the layers; each view co-owns the mmap (via owner_), so the
// returned Model is valid even after `g` is destroyed. Throws std::runtime_error
// with a clear message if a required tensor name is absent.
Model build_model_from_gguf(gguf::GGUF&& g);
