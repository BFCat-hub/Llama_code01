#include <device_launch_parameters.h>
#include <stdio.h>


__global__ void histogram(int n, const int* color, int* bucket) {
    int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    int num_threads = blockDim.x * gridDim.x;

    for (int i = 0; i < n; i += num_threads) {
        int c = color[i];
        atomicAdd(&bucket[c], 1);
    }
}

int main() {
    int n = 1000; 
    int num_buckets = 256;

    
    int* h_color = (int*)malloc(n * sizeof(int));
    int* h_bucket = (int*)malloc(num_buckets * sizeof(int));

    

    
    int* d_color, * d_bucket;
    cudaMalloc((void**)&d_color, n * sizeof(int));
    cudaMalloc((void**)&d_bucket, num_buckets * sizeof(int));

    
    cudaMemcpy(d_color, h_color, n * sizeof(int), cudaMemcpyHostToDevice);

    
    dim3 blockSize(256); 
    dim3 gridSize((n + blockSize.x - 1) / blockSize.x, 1);

    
    histogram<<<gridSize, blockSize>>>(n, d_color, d_bucket);

    
    cudaMemcpy(h_bucket, d_bucket, num_buckets * sizeof(int), cudaMemcpyDeviceToHost);

    

    
    free(h_color);
    free(h_bucket);
    cudaFree(d_color);
    cudaFree(d_bucket);

    return 0;
}