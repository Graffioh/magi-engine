#pragma once

struct ModelConfig {
    int   vocab_size        = 0;
    int   hidden_size       = 0;
    int   intermediate_size = 0;
    int   n_layers          = 0;
    int   n_heads           = 0;
    int   n_kv_heads        = 0;
    int   head_dim          = 0;
    int   max_seq_len       = 0;
    float rms_eps           = 1e-5f;
    int   rope_base         = 10000;
};
