#include <device_launch_parameters.h>
#include <stdio.h>


__global__ void Init(const long long size, const double* in, double* out) {
    long long i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < size) {
        out[i] = in[i];
    }
}

int main() {
    
    long long size = 512;

    
    double* h_in = (double*)malloc(size * sizeof(double));
    double* h_out = (double*)malloc(size * sizeof(double));

    
    for (long long i = 0; i < size; ++i) {
        h_in[i] = static_cast<double>(i);
    }

    
    double* d_in;
    double* d_out;
    cudaMalloc((void**)&d_in, size * sizeof(double));
    cudaMalloc((void**)&d_out, size * sizeof(double));

    
    cudaMemcpy(d_in, h_in, size * sizeof(double), cudaMemcpyHostToDevice);

    
    dim3 blockSize(256);
    dim3 gridSize((size + blockSize.x - 1) / blockSize.x, 1);

    
    Init<<<gridSize, blockSize>>>(size, d_in, d_out);

    
    cudaMemcpy(h_out, d_out, size * sizeof(double), cudaMemcpyDeviceToHost);

    
    for (long long i = 0; i < 10; ++i) {
        printf("%f ", h_out[i]);
    }

    
    free(h_in);
    free(h_out);
    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}