#include <cuda.h>
#include <stdio.h>

__global__
void matmulKernel(int n, float* d_a, float* d_b, float* d_c)
{
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;

        if (i < n && j  < n) {
                d_c[i*n + j] = 0;
                for (int k = 0; k < n; k++) {
                        d_c[i*n + j] += d_a[i*n + k] * d_b[k*n + j];
                }
        }
}

void matmul(int n, float* a, float* b, float* c)
{
        int size = n*n*sizeof(float);
        float *d_a; float *d_b; float *d_c;
        cudaMalloc((void **)&d_a, size);
        cudaMalloc((void **)&d_b, size);
        cudaMalloc((void **)&d_c, size);

        cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

        dim3 dimGrid(ceil(n / 16.0), ceil(n / 16.0), 1);
        dim3 dimBlock(16,16,1);
        matmulKernel<<<dimGrid, dimBlock>>> (n,d_a,d_b,d_c);
        cudaMemcpy(c,d_c,size, cudaMemcpyDeviceToHost);
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
}




int main(){

        printf("Enter size of square matrix : \n");
        int n;scanf("%d" , &n);
        printf("\nEnter elements of First Matrix : \n");
        float a[n][n];
        float b[n][n];
        float c[n][n];

        for(int i=0; i<n ; i++)
        {
                for(int j=0; j<n ; j++)
                {
                        scanf("%f", &a[i][j]);
                }
        }

        printf("\nEnter elements of Second Matrix : \n");
        for(int i=0; i<n ; i++)
        {
                for(int j=0; j<n ; j++)
                {
                        scanf("%f", &b[i][j]);
                }
        }

        matmul(n,(float*) a, (float*) b, (float*) c);
        printf("\nElements of Output Matrix : \n");
        for(int i=0; i<n ; i++)
        {
                for(int j=0; j<n ; j++)
                {
                        printf("%f ", c[i][j]);
                }
                printf("\n");
        }

}
