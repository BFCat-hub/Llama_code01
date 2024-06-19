#include <device_launch_parameters.h>
#include <stdio.h>


__global__ void histo(const unsigned int* const vals, unsigned int* const histo, size_t numVals) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numVals)
        return;
    
    atomicAdd(&histo[vals[i]], 1);
}

int main() {
    
    size_t numVals = 1000;

    
    unsigned int* h_vals = (unsigned int*)malloc(numVals * sizeof(unsigned int));
    unsigned int* h_histo = (unsigned int*)malloc(256 * sizeof(unsigned int));

    
    for (size_t i = 0; i < numVals; ++i) {
        h_vals[i] = i % 256; 
    }

    
    unsigned int* d_vals;
    unsigned int* d_histo;
    cudaMalloc((void**)&d_vals, numVals * sizeof(unsigned int));
    cudaMalloc((void**)&d_histo, 256 * sizeof(unsigned int));

    
    cudaMemcpy(d_vals, h_vals, numVals * sizeof(unsigned int), cudaMemcpyHostToDevice);

    
    dim3 blockSize(256);
    dim3 gridSize((numVals + blockSize.x - 1) / blockSize.x, 1);

    
    histo<<<gridSize, blockSize>>>(d_vals, d_histo, numVals);

    
    cudaMemcpy(h_histo, d_histo, 256 * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    
    for (size_t i = 0; i < 256; ++i) {
        printf("Histogram bin %d: %d occurrences\n", i, h_histo[i]);
    }

    
    free(h_vals);
    free(h_histo);
    cudaFree(d_vals);
    cudaFree(d_histo);

    return 0;
}