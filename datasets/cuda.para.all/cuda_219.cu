#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <stdio.h>

// CUDA核函数
__global__ void saxpi_nBlock(int n, float a, float *x, float *y) {
    int idx = threadIdx.x + (blockIdx.x * blockDim.x);
    if (idx < n) {
        y[idx] = a * x[idx] + y[idx];
    }
}

int main() {
    // 设置数组大小
    const int array_size = 100;

    // 在设备上分配空间
    float *x_device, *y_device;
    cudaMalloc((void**)&x_device, array_size * sizeof(float));
    cudaMalloc((void**)&y_device, array_size * sizeof(float));

    // 初始化输入数组数据
    float *x_host = (float *)malloc(array_size * sizeof(float));
    float *y_host = (float *)malloc(array_size * sizeof(float));
    for (int i = 0; i < array_size; ++i) {
        x_host[i] = i + 1.0; // 为了演示目的，将输入数据初始化为 1.0, 2.0, 3.0, ...
        y_host[i] = (i + 1.0) * 2.0; // 为了演示目的，将输入数据初始化为 2.0, 4.0, 6.0, ...
    }

    // 将输入数据从主机复制到设备
    cudaMemcpy(x_device, x_host, array_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(y_device, y_host, array_size * sizeof(float), cudaMemcpyHostToDevice);

    // 定义启动配置
    dim3 blockDim(256); // 块大小为256个线程
    dim3 gridDim((array_size + blockDim.x - 1) / blockDim.x); // 确保足够的块数

    // 定义标量值
    float a = 0.5;

    // 调用CUDA核函数
    saxpi_nBlock<<<gridDim, blockDim>>>(array_size, a, x_device, y_device);

    // 将结果从设备复制回主机
    float *y_result = (float *)malloc(array_size * sizeof(float));
    cudaMemcpy(y_result, y_device, array_size * sizeof(float), cudaMemcpyDeviceToHost);

    // 打印结果
    printf("Result after CUDA kernel execution:\n");
    for (int i = 0; i < array_size; ++i) {
        printf("%.2f ", y_result[i]);
    }
    printf("\n");

    // 释放内存
    free(x_host);
    free(y_host);
    free(y_result);
    cudaFree(x_device);
    cudaFree(y_device);

    return 0;
}
 
