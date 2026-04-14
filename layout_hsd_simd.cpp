constexpr int S = 1024;
constexpr int H = 16;
constexpr int D = 64;

// layout B: k_cache_B[h][s][d] (HND: n_heads, seq_len, head_dim)
alignas(64) float k_cache_B[H][S][D];

void attn_scores_layoutB(const float* __restrict q,
                        int h0,
                        float* __restrict out) {
  float (* __restrict k_head)[D] = k_cache_B[h0];

  for(int s = 0; s < S; ++s) {
    const float* __restrict k_row = k_head[s];
    float acc = 0.0f;
    
    #pragma clang loop vectorize(enable)
    for (int d = 0; d < D; ++d) {
      acc += q[d] * k_row[d];
    }
    out[s] = acc;
  }
}