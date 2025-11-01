#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define BD 256

__global__
void Red_Kernel(float *d_a, int n, float *d_ans)
{
        __shared__ float partial_sum[BD];
        int global_idx = blockDim.x * blockIdx.x + threadIdx.x;
        int idx = threadIdx.x;

        if (global_idx < n) {
                partial_sum[idx] = d_a[global_idx];
        } else {
                partial_sum[idx] = 0.0f;
        }
        __syncthreads();

        for(int i = blockDim.x / 2; i > 0 ; i /= 2)
        {
                if(idx < i)
                {
                        partial_sum[idx] += partial_sum[idx + i];
                }
                __syncthreads();
        }

        if(idx == 0)
        {
                atomicAdd(d_ans, partial_sum[0]);
        }
}

void reduce (int n, float *a, float *ans)
{
        int size = n * sizeof(float);
        float *d_a;
        float *d_ans;
        cudaMalloc((void **)&d_a, size);
        cudaMalloc((void **)&d_ans, sizeof(float));

        cudaMemset(d_ans, 0, sizeof(float));
        cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);

        int b_size = BD;
        dim3 BlockSize(b_size);
        dim3 GridSize((n + b_size - 1) / b_size);

        Red_Kernel<<<GridSize, BlockSize>>> (d_a, n, d_ans);

        cudaMemcpy(ans, d_ans, sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(d_a);
        cudaFree(d_ans);
}

int main()
{
        printf("Enter the size of array\n");
        int n;
        scanf("%d", &n);

        float *a = (float*)malloc(n * sizeof(float));

        printf("\nEnter elements of the array\n");
        for(int i = 0; i < n; i++) {
                scanf("%f", &a[i]);
        }

        float ans;
        reduce(n, a, &ans);
        printf("\nAns is : %f\n", ans);

        free(a);
        return 0;
}
