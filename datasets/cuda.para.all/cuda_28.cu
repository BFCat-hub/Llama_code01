#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void dsubtract_matrix(double* a, double* b, double* c, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        c[idx] = a[idx] - b[idx];
    }
}

int main() {
    // 设置数组大小
    int arraySize = 1000;

    // 分配主机端内存
    double* h_a = (double*)malloc(arraySize * sizeof(double));
    double* h_b = (double*)malloc(arraySize * sizeof(double));
    double* h_c = (double*)malloc(arraySize * sizeof(double));

    // 初始化数据
    for (int i = 0; i < arraySize; ++i) {
        h_a[i] = static_cast<double>(i);
        h_b[i] = static_cast<double>(2 * i);
    }

    // 分配设备端内存
    double* d_a;
    double* d_b;
    double* d_c;
    cudaMalloc((void**)&d_a, arraySize * sizeof(double));
    cudaMalloc((void**)&d_b, arraySize * sizeof(double));
    cudaMalloc((void**)&d_c, arraySize * sizeof(double));

    // 将数据从主机端拷贝到设备端
    cudaMemcpy(d_a, h_a, arraySize * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, arraySize * sizeof(double), cudaMemcpyHostToDevice);

    // 设置线程块和网格大小
    int blockSize = 256;
    int gridSize = (arraySize + blockSize - 1) / blockSize;

    // 启动内核
    dsubtract_matrix<<<gridSize, blockSize>>>(d_a, d_b, d_c, arraySize);

    // 将结果从设备端拷贝回主机端
    cudaMemcpy(h_c, d_c, arraySize * sizeof(double), cudaMemcpyDeviceToHost);

    // 打印结果（为了简化，此处不打印全部结果）
    for (int i = 0; i < 10; ++i) {
        printf("%f ", h_c[i]);
    }

    // 释放内存
    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
