#include <stdio.h>

void sgemm_kernelCPU(const float *host_inputArray1, const float *host_inputArray2, float *host_inputArray3, int M, int N, int K, float alpha, float beta);

int main() {
    // 在这里可以创建测试用的数据，并调用 sgemm_kernelCPU 函数
    // 例如：
    int M = 3;  // 你的矩阵维度
    int N = 4;
    int K = 5;

    // 假设 host_inputArray1、host_inputArray2 和 host_inputArray3 是相应大小的数组
    float host_inputArray1[M * K];
    float host_inputArray2[K * N];
    float host_inputArray3[M * N];

    // 初始化数组（这里只是示例，请根据你的实际需求初始化数据）
    for (int i = 0; i < M * K; i++) {
        host_inputArray1[i] = i + 1;
    }

    for (int i = 0; i < K * N; i++) {
        host_inputArray2[i] = i + 1;
    }

    for (int i = 0; i < M * N; i++) {
        host_inputArray3[i] = 0;
    }

    // 调用函数
    sgemm_kernelCPU(host_inputArray1, host_inputArray2, host_inputArray3, M, N, K, 1.0, 0.0);

    // 在这里可以添加打印结果或其他操作
    // 例如：
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%f ", host_inputArray3[i * N + j]);
        }
        printf("\n");
    }

    return 0;
}

void sgemm_kernelCPU(const float *host_inputArray1, const float *host_inputArray2, float *host_inputArray3, int M, int N, int K, float alpha, float beta) {
    for (int row = 0; row < M; row++) {
        for (int column = 0; column < N; column++) {
            float element_c = 0.f;
            for (int e = 0; e < K; e++) {
                element_c += host_inputArray1[row * K + e] * host_inputArray2[e * N + column];
            }
            host_inputArray3[row * N + column] = alpha * element_c + beta * host_inputArray3[row * N + column];
        }
    }
}
