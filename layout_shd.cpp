constexpr int S = 1024; // seq_len
constexpr int H = 16;   // n_heads
constexpr int D = 64;   // head_dim

// layout A: k_cache_A[s][d][h] (SDH: seq_len, head_dim, n_heads)
alignas(64) float k_cache_A[S][D][H];

void dot_good(
  const float* __restrict q,
  const float* __restrict k,
  float* __restrict out,
  int S, int H, int D, int head) {
    for(int s = 0; s < S; ++s) {
      const float* k_sh = k + (s * H + head) * D;

      float acc = 0.0f;

      #pragma clang loop vectorize(enable)
      for(int d = 0; d < D; ++d) {
        acc += q[d] * k_sh[d];
      }
      
      out[s] = acc;
    }
  }