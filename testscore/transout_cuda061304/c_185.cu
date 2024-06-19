#include <device_launch_parameters.h>
#include <stdio.h>


__global__ void setSuppressed(int* suppressed, int dims) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= dims) {
        return;
    }

    suppressed[tid] = 0;
}

int main() {
    int dims = 1000;

    int* h_suppressed = (int*)malloc(dims * sizeof(int));

    int* d_suppressed;
    cudaMalloc((void**)&d_suppressed, dims * sizeof(int));

    int blockSize = 256;
    int gridSize = (dims + blockSize - 1) / blockSize;

    setSuppressed<<<gridSize, blockSize>>>(d_suppressed, dims);

    cudaMemcpy(h_suppressed, d_suppressed, dims * sizeof(int), cudaMemcpyDeviceToHost);

    
    for (int i = 0; i < 10; ++i) {
        printf("%d ", h_suppressed[i]);
    }

    
    free(h_suppressed);
    cudaFree(d_suppressed);

    return 0;
}