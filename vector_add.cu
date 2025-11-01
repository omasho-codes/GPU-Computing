#include <cuda.h>
#include <iostream>
#include <vector>
using namespace std;

__global__
void vec_add_kernel(float *d_a, float *d_b, float *d_c, int n)
{
        int i = threadIdx.x + blockDim.x * blockIdx.x;
        if(i < n)  d_c[i] = d_a[i] + d_b[i];        
}

void vec_add(vector<float> &a, vector<float> &b, vector<float> &c, int n)
{
        int size = n * sizeof(float);
        float *d_a, *d_b, *d_c;

        cudaMalloc((void**)&d_a, size);
        cudaMalloc((void**)&d_b, size);
        cudaMalloc((void**)&d_c, size);

        cudaMemcpy(d_a, a.data(), size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, b.data(), size, cudaMemcpyHostToDevice);

        int blockSize = 256;
        int numBlocks = (n + blockSize - 1) / blockSize;
        vec_add_kernel<<<numBlocks, blockSize>>>(d_a, d_b, d_c, n);

        cudaMemcpy(c.data(), d_c, size, cudaMemcpyDeviceToHost);

        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
}

int main()
{
        int n;
        cout<<"Enter size of array: ";
        cin>>n;

        vector<float> a(n), b(n), c(n);

        cout<<"Enter elements of first array:\n";
        for(int i = 0; i < n; i++) cin>>a[i];

        cout<<"Enter elements of second array:\n";
        for(int i = 0; i < n; i++) cin>>b[i];

        vec_add(a, b, c, n);

        cout<<"\nResultant array:\n";
        for(int i = 0; i < n; i++) cout<<c[i]<<" ";
        cout<<"\n";
}
