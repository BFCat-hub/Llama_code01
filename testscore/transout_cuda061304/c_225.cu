#include <device_launch_parameters.h>
#include <stdio.h>


__global__ void addKernel(int* result, const int* val1, const int* val2, size_t size) {
    int gid = blockDim.x * blockIdx.x + threadIdx.x;
    if (gid < size) {
        result[gid] = val1[gid] + val2[gid];
    }
}

int main() {
    
    size_t arraySize = 1000;

    
    int* h_result = (int*)malloc(arraySize * sizeof(int));
    int* h_val1 = (int*)malloc(arraySize * sizeof(int));
    int* h_val2 = (int*)malloc(arraySize * sizeof(int));

    
    for (size_t i = 0; i < arraySize; ++i) {
        h_val1[i] = i;
        h_val2[i] = 2 * i;
    }

    
    int* d_result;
    int* d_val1;
    int* d_val2;
    cudaMalloc((void**)&d_result, arraySize * sizeof(int));
    cudaMalloc((void**)&d_val1, arraySize * sizeof(int));
    cudaMalloc((void**)&d_val2, arraySize * sizeof(int));

    
    cudaMemcpy(d_val1, h_val1, arraySize * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_val2, h_val2, arraySize * sizeof(int), cudaMemcpyHostToDevice);

    
    int blockSize = 256;
    int gridSize = (arraySize + blockSize - 1) / blockSize;

    
    addKernel<<<gridSize, blockSize>>>(d_result, d_val1, d_val2, arraySize);

    
    cudaMemcpy(h_result, d_result, arraySize * sizeof(int), cudaMemcpyDeviceToHost);

    
    for (size_t i = 0; i < 10; ++i) {
        printf("%d ", h_result[i]);
    }

    
    free(h_result);
    free(h_val1);
    free(h_val2);
    cudaFree(d_result);
    cudaFree(d_val1);
    cudaFree(d_val2);

    return 0;
}