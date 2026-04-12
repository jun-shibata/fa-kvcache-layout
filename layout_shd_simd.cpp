constexpr int S = 1024; // seq_len
constexpr int H = 16;   // n_heads
constexpr int D = 64;   // head_dim
constexpr int BLOCK_S = 128; // seq block size

// layout A: k_cache_A[s][h][d] (NHD: seq_len, n_heads, head_dim)
alignas(64) float k_cache_A[S][H][D];

void attn_scores_layoutA(const float* __restrict q,
                        int h0,
                        float* __restrict out) {
  for (int s = 0; s < S; ++s) {
    float acc = 0.0f;
    for (int d = 0; d < D; ++d) {
      float k = k_cache_A[s][h0][d];
      acc += q[d] * k;
    }
    out[s] = acc;
  }
}