#include <device_launch_parameters.h>
#include <stdio.h>


#define N 16

__global__ void vecAddGPU(float* input_a, float* input_b, float* output) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    output[idx] = input_a[idx] + input_b[idx];
}

int main() {
    
    float h_input_a[N];
    float h_input_b[N];
    float h_output[N];

    
    for (int i = 0; i < N; ++i) {
        h_input_a[i] = static_cast<float>(i);
        h_input_b[i] = static_cast<float>(2 * i);
    }

    
    float* d_input_a;
    float* d_input_b;
    float* d_output;
    cudaMalloc((void**)&d_input_a, N * sizeof(float));
    cudaMalloc((void**)&d_input_b, N * sizeof(float));
    cudaMalloc((void**)&d_output, N * sizeof(float));

    
    cudaMemcpy(d_input_a, h_input_a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input_b, h_input_b, N * sizeof(float), cudaMemcpyHostToDevice);

    
    dim3 blockSize(256);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, 1);

    
    vecAddGPU<<<gridSize, blockSize>>>(d_input_a, d_input_b, d_output);

    
    cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);

    
    for (int i = 0; i < 8; ++i) {
        printf("h_output[%d]: %f\n", i, h_output[i]);
    }

    
    cudaFree(d_input_a);
    cudaFree(d_input_b);
    cudaFree(d_output);

    return 0;
}

```