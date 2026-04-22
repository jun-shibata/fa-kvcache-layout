#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <cuda_runtime.h>

using namespace std;

// --- layout: k[s][h][d] ---

__global__ void dot_shd_kernel(
  const float* q, // [D]
  const float* k, // [S][H][D]
  float* out,     // [S]
  int S, int H, int D, int head) {
  int s = blockIdx.x * blockDim.x + threadIdx.x;
  if (s >= S) return;

  const float* k_sh = k + (s * H + head) * D;

  float acc = 0.0f;
  for(int d = 0; d < D; ++d) {
    acc += q[d] * k_sh[d];
  }
  out[s] = acc;
}

// --- layout: k[d][s][h] ---

__global__ void dot_dsh_kernel(
  const float* q, // [D]
  const float* k, // [D][S][H]
  float* out,     // [S]
  int S, int H, int D, int head) {
  int s = blockIdx.x * blockDim.x + threadIdx.x;
  if (s >= S) return;
  
  float acc = 0.0f;
  for(int d = 0; d < D; ++d) {
    const float k_val = k[(d * S + s) * H + head];
    acc += q[d] * k_val;
  }
  out[s] = acc;
}

// --- initialize ---

void init(vector<float>& v) {
  std::mt19937 rng(0);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for(auto& x: v) x = dist(rng);
}

// --- benchmark ---

template<typename F>
double bench_cuda(F func, int iters = 1000) {
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  
  float total_time = 0.0f;
  for(int i = 0; i < iters; ++i) {
    cudaEventRecord(start);
    func();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    total_time += milliseconds;
  }
  
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return total_time / iters;
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

  // Allocate GPU memory
  float *d_q, *d_k_shd, *d_k_dsh, *d_out;
  cudaMalloc(&d_q, D * sizeof(float));
  cudaMalloc(&d_k_shd, S * H * D * sizeof(float));
  cudaMalloc(&d_k_dsh, D * S * H * sizeof(float));
  cudaMalloc(&d_out, S * sizeof(float));

  // Copy data to GPU
  cudaMemcpy(d_q, q.data(), D * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_k_shd, k_shd.data(), S * H * D * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_k_dsh, k_dsh.data(), D * S * H * sizeof(float), cudaMemcpyHostToDevice);

  // Kernel launch parameters
  int blockSize = 256;
  int numBlocks = (S + blockSize - 1) / blockSize;

  // Create CUDA events for timing
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Warm up and verify SHD
  dot_shd_kernel<<<numBlocks, blockSize>>>(d_q, d_k_shd, d_out, S, H, D, head);
  cudaDeviceSynchronize();
  cudaMemcpy(out.data(), d_out, S * sizeof(float), cudaMemcpyDeviceToHost);
  cout << "out[0]: " << out[0] << endl;

  // Warm up and verify DSH
  dot_dsh_kernel<<<numBlocks, blockSize>>>(d_q, d_k_dsh, d_out, S, H, D, head);
  cudaDeviceSynchronize();
  cudaMemcpy(out.data(), d_out, S * sizeof(float), cudaMemcpyDeviceToHost);
  cout << "out[0]: " << out[0] << endl;

  // Benchmark SHD
  double t_shd = bench_cuda([&]() {
    dot_shd_kernel<<<numBlocks, blockSize>>>(d_q, d_k_shd, d_out, S, H, D, head);
  }, 1000);
  cout << "Layout SHD : " << t_shd << " ms" << endl;

  // Benchmark DSH
  double t_dsh = bench_cuda([&]() {
    dot_dsh_kernel<<<numBlocks, blockSize>>>(d_q, d_k_dsh, d_out, S, H, D, head);
  }, 1000);
  cout << "Layout DSH : " << t_dsh << " ms" << endl;
  cudaFree(d_q);
  cudaFree(d_k_shd);
  cudaFree(d_k_dsh);
  cudaFree(d_out);

  return 0;
}