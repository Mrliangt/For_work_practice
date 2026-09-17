/**
 * @file test
 * @author tian
 * @date 2026/9/3
 */

#if 0
// Attention = softmax(QKT/ sqrt(dk)) V
// Q = XWT;[128,1024] * [1024, 128]
//

void QK() {
    int group = Q[:3] / head_dim;
    vector<float> score;
    for (i = 0; i < group; ++i) {
        score.push_back(softmax(Q[::i * 128] * K)V);
    }
    [1, 50]
}

// glu = output * Wt + gate(selu(output));
// 量化方式

__device__ void reducesum(const float* x, float& sum, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float t_sum = 0.0f;

//    if (tid < N) {
//        x[tid] += x[tid + blockDim.x];
//    }
//    __syncthreads();
    int warp_id = tid / 32;
    int warp_tid = tid % 32;
    // warp内
    __shared__ float sum[warp];
    __shuffle__
    for (int i = BlockDim.x / 2; i > 0; i >> 1) {
        x[] += // block块内规约
    }
    __syncthreads();
    if (tid == 0) {
        sum = x[0];
    }
}

__global__ void softmax(const float* x) {

}

// vllm\sglang-mini\nsight\online softmax\
#endif