#include <device_launch_parameters.h>
#include <stdio.h>


__global__ void kmeans_set_zero(int* means, int size) {
    int id = threadIdx.x + blockDim.x * blockIdx.x;
    int num_blocks = (size + blockDim.x - 1) / blockDim.x;

    if (id < size) {
        means[id] = 0;
    }
}

int main() {
    int size = 1000; 

    int* h_means = (int*)malloc(size * sizeof(int));

    int* d_means;
    cudaMalloc((void**)&d_means, size * sizeof(int));

    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;

    kmeans_set_zero<<<gridSize, blockSize>>>(d_means, size);

    cudaMemcpy(h_means, d_means, size * sizeof(int), cudaMemcpyDeviceToHost);

    
    for (int i = 0; i < 10; ++i) {
        printf("%d ", h_means[i]);
    }

    
    free(h_means);
    cudaFree(d_means);

    return 0;
}