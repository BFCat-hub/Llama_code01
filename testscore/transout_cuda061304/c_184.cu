#include <device_launch_parameters.h>
#include <stdio.h>


__global__ void set_offset_kernel(int stride, int size, int* output) {
    int thread_index = threadIdx.x + blockIdx.x * blockDim.x;
    int num_threads = blockDim.x * gridDim.x;

    for (int i = 0; i < size; i++) {
        int element_index = i * num_threads + thread_index;

        if (element_index < size) {
            output[element_index] = i * stride;
        }
    }
}

int main() {
    int stride = 2; 
    int size = 100; 

    int* h_output = (int*)malloc(size * sizeof(int));

    int* d_output;
    cudaMalloc((void**)&d_output, size * sizeof(int));

    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;

    set_offset_kernel<<<gridSize, blockSize>>>(stride, size, d_output);

    cudaMemcpy(h_output, d_output, size * sizeof(int), cudaMemcpyDeviceToHost);

    
    for (int i = 0; i < 10; ++i) {
        printf("%d ", h_output[i]);
    }

    
    free(h_output);
    cudaFree(d_output);

    return 0;
}