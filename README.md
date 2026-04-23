# fa-kvcache-layout
Sample implementations for comparing FlashAttention-optimized KV-cache layouts with standard layouts.

```shell
$ clang++ -O3 -ffast-math -march=native -Rpass=loop-vectorize benchmark_layout_shd_dsh.cpp -o benchmark_layout_shd_dsh.out
$ ./benchmark_layout_shd_dsh.out
```

There is also a code implementation for comparing layouts in CUDA.
