#include <device_launch_parameters.h>
#include <stdio.h>


__global__ void kernelUpdateHead(int* head, int* d_idxs_out, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < n) {
        head[d_idxs_out[tid]] = 1;
    }
}

int main() {
    int n = 1000; 

    int* h_head = (int*)malloc(n * sizeof(int));
    int* h_d_idxs_out = (int*)malloc(n * sizeof(int));

    int* d_head, * d_d_idxs_out;
    cudaMalloc((void**)&d_head, n * sizeof(int));
    cudaMalloc((void**)&d_d_idxs_out, n * sizeof(int));

    cudaMemcpy(d_head, h_head, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_d_idxs_out, h_d_idxs_out, n * sizeof(int), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    kernelUpdateHead<<<gridSize, blockSize>>>(d_head, d_d_idxs_out, n);

    cudaMemcpy(h_head, d_head, n * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 10; ++i) {
        printf("h_head[%d]: %d\n", i, h_head[i]);
    }

    
    free(h_head);
    free(h_d_idxs_out);
    cudaFree(d_head);
    cudaFree(d_d_idxs_out);

    return 0;
}