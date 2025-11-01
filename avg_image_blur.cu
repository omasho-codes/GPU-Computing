#include <stdio.h>
#include <cuda_runtime.h>

__global__
void blurKernel(int n, int p, float* d_a, float* d_b)
{
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;

        if (i < n && j < n) {
                float sum = 0.0f;
                int pixels = 0;
                int half = p / 2;

                for (int p_i = -half; p_i <= half; p_i++) {
                        int ni = i + p_i;
                        if (ni >= 0 && ni < n) {
                                for (int p_j = -half; p_j <= half; p_j++) {
                                        int nj = j + p_j;
                                        if (nj >= 0 && nj < n) {
                                                sum += d_a[ni * n + nj];
                                                pixels++;
                                        }
                                }
                        }
                }
                d_b[i * n + j] = sum / pixels;
        }
}

void blurMatrix(int n, int p, float* a, float* b)
{
        int size = n * n * sizeof(float);
        float *d_a, *d_b;

        cudaMalloc((void**)&d_a, size);
        cudaMalloc((void**)&d_b, size);

        cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);

        dim3 dimBlock(16, 16, 1);
        dim3 dimGrid((n + 15) / 16, (n + 15) / 16, 1);

        blurKernel<<<dimGrid, dimBlock>>>(n, p, d_a, d_b);

        cudaMemcpy(b, d_b, size, cudaMemcpyDeviceToHost);

        cudaFree(d_a);
        cudaFree(d_b);
}

int main()
{
        int n, p;
        printf("Enter size of square matrix: ");
        scanf("%d", &n);

        printf("Enter blur window size (odd integer): ");
        scanf("%d", &p);

        float a[n][n], b[n][n];

        printf("\nEnter elements of input matrix:\n");
        for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                        scanf("%f", &a[i][j]);
                }
        }

        blurMatrix(n, p, (float*)a, (float*)b);

        printf("\nBlurred output matrix:\n");
        for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                        printf("%6.2f ", b[i][j]);
                }
                printf("\n");
        }

        return 0;
