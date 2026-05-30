#pragma once

#include <vector>

class RopeCache {
  private:
    std::vector<float> cos_cache_;
    std::vector<float> sin_cache_;
  public:
    RopeCache(const int max_seq_len, const int head_dim, const int base = 10000);

    const std::vector<float>& cos_cache() const { return cos_cache_; }

    const std::vector<float>& sin_cache() const { return sin_cache_; }
};
