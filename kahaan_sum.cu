#include <stdio.h>
#include <cuda.h>

__device__ void kahanSum(float *sum, float *c, float input) {
        float y = input - *c;
        float t = *sum + y;
        *c = (t - *sum) - y;
        *sum = t;
}

__global__ void kahanSumKernel(float *input, float *output, int n) {
        extern __shared__ float shared[];

        int tid = threadIdx.x;
        int idx = blockIdx.x * blockDim.x + tid;

        float sum = 0.0f;
        float c = 0.0f;

        for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
                kahanSum(&sum, &c, input[i]);
        }

        shared[tid] = sum;
        __syncthreads();

        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
                if (tid < s) {
                        shared[tid] += shared[tid + s];
                }
                __syncthreads();
        }

        if (tid == 0) {
                output[blockIdx.x] = shared[0];
        }
}

int main() {
        const int N = 1 << 20;
        float *h_input = new float[N];

        for (int i = 0; i < N; i++) {
                h_input[i] = 1.0f / (i + 1);
        }

        float *d_input, *d_partialSums;
        cudaMalloc(&d_input, N * sizeof(float));

        int blocks = 256;
        int threads = 256;
        cudaMalloc(&d_partialSums, blocks * sizeof(float));

        cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

        kahanSumKernel<<<blocks, threads, threads * sizeof(float)>>>(d_input, d_partialSums, N);

        float *h_partialSums = new float[blocks];
        cudaMemcpy(h_partialSums, d_partialSums, blocks * sizeof(float), cudaMemcpyDeviceToHost);

        float sum = 0.0f;
        float c = 0.0f;
        for (int i = 0; i < blocks; i++) {
                float y = h_partialSums[i] - c;
                float t = sum + y;
                c = (t - sum) - y;
                sum = t;
        }

        printf("Kahan Summation result: %.10f\n", sum);

        delete[] h_input;
        delete[] h_partialSums;
        cudaFree(d_input);
        cudaFree(d_partialSums);

        return 0;
}
