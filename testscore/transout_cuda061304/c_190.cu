#include <device_launch_parameters.h>
#include <stdio.h>


__global__ void subAvg(int* input, int count, int avg) {
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    int stride = blockDim.x * gridDim.x;

    while (index < count) {
        input[index] -= avg;
        index += stride;
    }
}

int main() {
    
    int count = 512;
    int avg = 0;

    
    int* h_input = (int*)malloc(count * sizeof(int));

    
    for (int i = 0; i < count; ++i) {
        h_input[i] = i;
    }

    
    int* d_input;
    cudaMalloc((void**)&d_input, count * sizeof(int));

    
    cudaMemcpy(d_input, h_input, count * sizeof(int), cudaMemcpyHostToDevice);

    
    int blockSize = 256;
    int gridSize = (count + blockSize - 1) / blockSize;

    
    subAvg<<<gridSize, blockSize>>>(d_input, count, avg);

    
    cudaMemcpy(h_input, d_input, count * sizeof(int), cudaMemcpyDeviceToHost);

    
    for (int i = 0; i < 10; ++i) {
        printf("%d ", h_input[i]);
    }

    
    free(h_input);
    cudaFree(d_input);

    return 0;
}