#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <stdio.h>

// CUDA核函数
__global__ void setSuppressed(int *suppressed, int dims) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= dims) {
        return;
    }
    suppressed[tid] = 0;
}

int main() {
    // 设置数据大小
    const int data_size = 100;

    // 在设备上分配空间
    int *suppressed_device;
    cudaMalloc((void**)&suppressed_device, data_size * sizeof(int));

    // 定义启动配置
    dim3 blockDim(256); // 块大小为256个线程
    dim3 gridDim((data_size + blockDim.x - 1) / blockDim.x); // 确保足够的块数

    // 调用CUDA核函数
    setSuppressed<<<gridDim, blockDim>>>(suppressed_device, data_size);

    // 将结果从设备复制回主机
    int *suppressed_host = (int *)malloc(data_size * sizeof(int));
    cudaMemcpy(suppressed_host, suppressed_device, data_size * sizeof(int), cudaMemcpyDeviceToHost);

    // 打印结果
    printf("Result after CUDA kernel execution:\n");
    for (int i = 0; i < data_size; ++i) {
        printf("%d ", suppressed_host[i]);
    }
    printf("\n");

    // 释放内存
    free(suppressed_host);
    cudaFree(suppressed_device);

    return 0;
}
 
