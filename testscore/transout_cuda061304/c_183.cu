#include <device_launch_parameters.h>
#include <stdio.h>


__global__ void resetIndices(long* vec_out, const long N) {
    long idx = threadIdx.x + blockDim.x * blockIdx.x;
    while (idx < N) {
        vec_out[idx] = idx;
        idx += blockDim.x * gridDim.x;
    }
}

int main() {
    
    long N = 1000;

    
    long* h_vec_out = (long*)malloc(N * sizeof(long));

    
    long* d_vec_out;
    cudaMalloc((void**)&d_vec_out, N * sizeof(long));

    
    dim3 gridSize((N + 255) / 256, 1, 1);
    dim3 blockSize(256, 1, 1);

    
    resetIndices<<<gridSize, blockSize>>>(d_vec_out, N);

    
    cudaMemcpy(h_vec_out, d_vec_out, N * sizeof(long), cudaMemcpyDeviceToHost);

    
    for (int i = 0; i < 10; ++i) {
        printf("%ld ", h_vec_out[i]);
    }

    
    free(h_vec_out);
    cudaFree(d_vec_out);

    return 0;
}