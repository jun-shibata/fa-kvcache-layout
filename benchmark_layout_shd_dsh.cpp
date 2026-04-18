#include <iostream>
#include <vector>
#include <chrono>
#include <random>

using namespace std;

// --- layout: k[s][h][d] ---

void dot_shd(
  const float* q, // [D]
  const float* k, // [S][H][D]
  float* out,     // [S]
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

// --- layout: k[d][s][h] ---

void dot_dsh(
  const float* q, // [D]
  const float* k, // [D][S][H]
  float* out,     // [S]
  int S, int H, int D, int head) {
  for(int s = 0; s < S; ++s) {
    
    float acc = 0.0f;
    
    #pragma clang loop vectorize(enable)
    for(int d = 0; d < D; ++d) {
      const float k_val = k[(d * S + s) * H + head];
      acc += q[d] * k_val;
    }
    out[s] = acc;
  }
}

// --- initialize ---

void init(vector<float>& v) {
  std::mt19937 rng(0);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for(auto& x: v) x = dist(rng);
}

// --- benchmark ---

template<typename F>
double bench(F func, int iters = 10000) {
  auto start = chrono::high_resolution_clock::now();
  for(int i = 0; i < iters; ++i) {
    func();
  }
  auto end = chrono::high_resolution_clock::now();
  return chrono::duration<double, milli>(end - start).count();
}

int main() {
  constexpr int S = 1024; // seq_len
  constexpr int H = 16;   // n_heads
  constexpr int D = 64;   // head_dim
  const int head = 3;

  vector<float> q(D);
  vector<float> k_shd(S * H * D);
  vector<float> k_dsh(D * S * H);
  vector<float> out(S);

  init(q);
  init(k_shd);
  init(k_dsh);

  dot_shd(q.data(), k_shd.data(), out.data(), S, H, D, head);
  dot_dsh(q.data(), k_dsh.data(), out.data(), S, H, D, head);
  
  double t_shd = bench([&]() {
    dot_shd(q.data(), k_shd.data(), out.data(), S, H, D, head);
  });
  cout << "out[0]: " << out[0] << endl;

  double t_dsh = bench([&]() {
    dot_dsh(q.data(), k_dsh.data(), out.data(), S, H, D, head);
  });
  cout << "out[0]: " << out[0] << endl;

  cout << "Layout SHD : " << t_shd << " ms" << endl;
  cout << "Layout DSH : " << t_dsh << " ms" << endl;
  return 0;
}