#include <device_launch_parameters.h>
#include <stdio.h>


__global__ void allDivInplaceKernel(double* arrid, double scalar, int n) {
    int tx = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tx;
    int numThreads = blockDim.x * gridDim.x;

    for (int i = 0; i < n; i += numThreads) {
        int index = i + idx;

        if (index < n) {
            arrid[index] /= scalar;
        }
    }
}

int main() {
    int n = 1000;
    double scalar = 2.0;

    double* h_arrid = (double*)malloc(n * sizeof(double));

    for (int i = 0; i < n; ++i) {
        h_arrid[i] = static_cast<double>(i);
    }

    double* d_arrid;
    cudaMalloc((void**)&d_arrid, n * sizeof(double));

    cudaMemcpy(d_arrid, h_arrid, n * sizeof(double), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    allDivInplaceKernel<<<gridSize, blockSize>>>(d_arrid, scalar, n);

    cudaMemcpy(h_arrid, d_arrid, n * sizeof(double), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 10; ++i) {
        printf("%f ", h_arrid[i]);
    }

    free(h_arrid);
    cudaFree(d_arrid);

    return 0;
}