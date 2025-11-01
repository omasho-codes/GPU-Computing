#include <stdio.h>
#include <cuda_runtime.h>

#define CHECK(call) { \
        const cudaError_t error = call; \
        if (error != cudaSuccess) { \
                fprintf(stderr, "Error: %s:%d, code:%d, reason: %s\n", \
                                __FILE__, __LINE__, error, cudaGetErrorString(error)); \
                exit(1); \
        } \
}

__global__ void kernel(float *F, double *D)
{
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid == 0)
        {
                *F = 128.0f + 1e-6f;
                *D = 128.0  + 1e-6;
        }
}

int main()
{
        float *d_F, h_F;
        double *d_D, h_D;

        CHECK(cudaMalloc((void**)&d_F, sizeof(float)));
        CHECK(cudaMalloc((void**)&d_D, sizeof(double)));

        kernel<<<1, 1>>>(d_F, d_D);
        CHECK(cudaDeviceSynchronize());

        CHECK(cudaMemcpy(&h_F, d_F, sizeof(float), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(&h_D, d_D, sizeof(double), cudaMemcpyDeviceToHost));

        printf("Single precision result (float) : %.10f\n", h_F);
        printf("Double precision result (double): %.15lf\n", h_D);

        cudaFree(d_F);
        cudaFree(d_D);
        return 0;
}
