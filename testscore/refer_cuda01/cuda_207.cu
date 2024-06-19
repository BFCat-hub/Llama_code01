#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <stdio.h>

// CUDA核函数
__global__ void dadd_matrix(double *a, double *b, double *c, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    // 设置矩阵大小
    const int matrix_size = 100;

    // 在设备上分配空间
    double *a_device, *b_device, *c_device;
    cudaMalloc((void**)&a_device, matrix_size * sizeof(double));
    cudaMalloc((void**)&b_device, matrix_size * sizeof(double));
    cudaMalloc((void**)&c_device, matrix_size * sizeof(double));

    // 初始化输入矩阵数据
    double *a_host = (double *)malloc(matrix_size * sizeof(double));
    double *b_host = (double *)malloc(matrix_size * sizeof(double));
    for (int i = 0; i < matrix_size; ++i) {
        a_host[i] = i + 1.0; // 为了演示目的，将输入数据初始化为 1.0, 2.0, 3.0, ...
        b_host[i] = (i + 1.0) * 2.0; // 为了演示目的，将输入数据初始化为 2.0, 4.0, 6.0, ...
    }

    // 将输入数据从主机复制到设备
    cudaMemcpy(a_device, a_host, matrix_size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(b_device, b_host, matrix_size * sizeof(double), cudaMemcpyHostToDevice);

    // 定义启动配置
    dim3 blockDim(256); // 块大小为256个线程
    dim3 gridDim((matrix_size + blockDim.x - 1) / blockDim.x); // 确保足够的块数

    // 调用CUDA核函数
    dadd_matrix<<<gridDim, blockDim>>>(a_device, b_device, c_device, matrix_size);

    // 将结果从设备复制回主机
    double *c_result = (double *)malloc(matrix_size * sizeof(double));
    cudaMemcpy(c_result, c_device, matrix_size * sizeof(double), cudaMemcpyDeviceToHost);

    // 打印结果
    printf("Result after CUDA kernel execution:\n");
    for (int i = 0; i < matrix_size; ++i) {
        printf("%.2f ", c_result[i]);
    }
    printf("\n");

    // 释放内存
    free(a_host);
    free(b_host);
    free(c_result);
    cudaFree(a_device);
    cudaFree(b_device);
    cudaFree(c_device);

    return 0;
}
 
