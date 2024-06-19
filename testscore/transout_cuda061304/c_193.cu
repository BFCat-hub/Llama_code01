#include <device_launch_parameters.h>
#include <stdio.h>


__global__ void setLabels(int* output, int dims, int clsNum) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < dims) {
        output[tid] = tid % clsNum;
    }
}

int main() {
    
    int dims = 1000;
    int clsNum = 10;

    
    int* h_output = (int*)malloc(dims * sizeof(int));

    
    int* d_output;
    cudaMalloc((void**)&d_output, dims * sizeof(int));

    
    int blockSize = 256;
    int gridSize = (dims + blockSize - 1) / blockSize;

    
    setLabels<<<gridSize, blockSize>>>(d_output, dims, clsNum);

    
    cudaMemcpy(h_output, d_output, dims * sizeof(int), cudaMemcpyDeviceToHost);

    
    for (int i = 0; i < 10; ++i) {
        printf("%d ", h_output[i]);
    }

    
    free(h_output);
    cudaFree(d_output);

    return 0;
}