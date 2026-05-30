#include "utils.h"

#include <cmath>

RopeCache::RopeCache(const int max_seq_len, const int head_dim, const int base) {
    int head_dim_half = head_dim / 2;

    cos_cache_.resize(max_seq_len * (head_dim_half));
    sin_cache_.resize(max_seq_len * (head_dim_half));

    for (int s = 0; s < max_seq_len; ++s) {
        for (int p = 0; p < head_dim / 2; ++p) {
            const float freq  = powf(base, (-2.0f * p / head_dim));
            const float theta = s * freq;

            cos_cache_[s * head_dim_half + p] = cosf(theta);
            sin_cache_[s * head_dim_half + p] = sinf(theta);
        }
    }
}
