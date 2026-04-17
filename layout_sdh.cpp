constexpr int S = 1024; // seq_len
constexpr int H = 16;   // n_heads
constexpr int D = 64;   // head_dim

// layout A: k_cache_A[s][d][h] (SDH: seq_len, head_dim, n_heads)
alignas(64) float k_cache_A[S][D][H];

void attn_dot_layout_bad_stride(const float* __restrict q,
                                int h0,
                                float* __restrict out) {
  for (int s = 0; s < S; ++s) {
    float acc = 0.0f;

    #pragma clang loop vectorize(enable)
    for (int d = 0; d < D; ++d) {
      float k = k_cache_A[s][d][h0];
      acc += q[d] * k;
    }
    out[s] = acc;
  }
}