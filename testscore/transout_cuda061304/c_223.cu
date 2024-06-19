#include <device_launch_parameters.h>
#include <stdio.h>

__global__ void vectorAdd(float* array, int maxElements, int numElements) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    int i = 0;
    while (index < numElements) {
        array[index] += array[index];
        index += stride;
        i++;
    }
}

int main() {
    
    int maxElements = 1000;
    int numElements = 500;

    
    float* h_array = (float*)malloc(maxElements * sizeof(float));

    
    for (int i = 0; i < numElements; i++) {
        h_array[i] = static_cast<float>(i);
    }

    
    float* d_array;
    cudaMalloc((void**)&d_array, maxElements * sizeof(float));

    
    cudaMemcpy(d_array, h_array, maxElements * sizeof(float), cudaMemcpyHostToDevice);

    
    dim3 blockSize(256);
    dim3 gridSize((numElements + blockSize.x - 1) / blockSize.x, 1);

    
    vectorAdd<<<gridSize, blockSize>>>(d_array, maxElements, numElements);

    
    cudaMemcpy(h_array, d_array, maxElements * sizeof(float), cudaMemcpyDeviceToHost);

    
    for (int i = 0; i < numElements; i++) {
        printf("%f ", h_array[i]);
    }

    
    free(h_array);
    cudaFree(d_array);

    return 0;
}