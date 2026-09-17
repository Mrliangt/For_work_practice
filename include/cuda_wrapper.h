/**
 * @file cuda_wrapper
 * @author tian
 * @date 2026/8/21
 */

#include <cuda_runtime.h>

__device__ float warp_reduce_sum(float val);
__device__ int warp_reduce_sum(int val);
__global__ void test_warp_reduce();
__global__ void block_reduce_kernel(float* input, float* output, int N);
__global__ void reduce_sum_v0(float* input, float* output, int N);
__global__ void reduce_sum_v1(float* input, float* output, int N);
__global__ void reduce_sum_v2(float* input, float* output, int N);
__global__ void reduce_sum_v3(float* input, float* output, int N);

// GEMM实现
void cpu_gemm(const float* A, const float* B, float* C, int M, int N, int K);
__global__ void gpu_gemm_v1(const float* A, const float* B, float* C, int M, int N, int K);
__global__ void gpu_gemm_v2(const float* A, const float* B, float* C, int M, int N, int K);
__global__ void gpu_gemm_v3(const float* A, const float* B, float* C, int M, int N, int K);

// SoftMax实现
__global__ void softmax_v0(float* input, float* output, int M, int N);
__global__ void softmax_v1(float* input, float* output, int M, int N);
__global__ void softmax_v2(float* input, float* output, int M, int N);