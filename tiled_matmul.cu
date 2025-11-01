#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

#define BW 16

__global__
void matmulKernel(int n, float* d_a, float* d_b, float* d_c)
{
        __shared__ float mds[BW][BW];
        __shared__ float nds[BW][BW];

        int tx = threadIdx.x;
        int ty = threadIdx.y;

        int row = blockIdx.y * blockDim.y + ty;
        int col = blockIdx.x * blockDim.x + tx;

        float sum = 0.0f;

        for (int m = 0; m < (n + BW - 1) / BW; m++)
        {
                int a_row = row;
                int a_col = m * BW + tx;

                if (a_row < n && a_col < n) {
                        mds[ty][tx] = d_a[a_row * n + a_col];
                } else {
                        mds[ty][tx] = 0.0f;
                }

                int b_row = m * BW + ty;
                int b_col = col;

                if (b_row < n && b_col < n) {
                        nds[ty][tx] = d_b[b_row * n + b_col];
                } else {
                        nds[ty][tx] = 0.0f;
                }

                __syncthreads();

                for (int k = 0; k < BW; k++) {
                        sum += mds[ty][k] * nds[k][tx];
                }

                __syncthreads();
        }

        if (row < n && col < n) {
                d_c[row * n + col] = sum;
        }
}

void matmul(int n, float* a, float* b, float* c)
{
        int size = n * n * sizeof(float);
        float *d_a, *d_b, *d_c;

        cudaMalloc((void **)&d_a, size);
        cudaMalloc((void **)&d_b, size);
        cudaMalloc((void **)&d_c, size);

        cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

        dim3 dimGrid(ceil(n / (float)BW), ceil(n / (float)BW), 1);
        dim3 dimBlock(BW, BW, 1);

        matmulKernel<<<dimGrid, dimBlock>>>(n, d_a, d_b, d_c);

        cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
}

int main() {
        printf("Enter size of square matrix : \n");
        int n;
        scanf("%d", &n);

        float *a = (float*)malloc(n * n * sizeof(float));
        float *b = (float*)malloc(n * n * sizeof(float));
        float *c = (float*)malloc(n * n * sizeof(float));

        printf("\nEnter elements of First Matrix : \n");
        for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                        scanf("%f", &a[i * n + j]);
                }
        }

        printf("\nEnter elements of Second Matrix : \n");
        for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                        scanf("%f", &b[i * n + j]);
                }
        }

        matmul(n, a, b, c);

        printf("\nElements of Output Matrix : \n");
        for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                        printf("%f ", c[i * n + j]);
                }
                printf("\n");
        }

        free(a);
        free(b);
        free(c);

        return 0;
}
