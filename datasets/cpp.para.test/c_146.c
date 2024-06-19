#include <stdio.h>

void mul(float *M, float *N, float *K, float height_M, float width_N, float width_M);

int main() {
    // 在这里可以创建测试用的数据，并调用 mul 函数
    // 例如：
    float height_M = 3;
    float width_N = 4;
    float width_M = 2;

    // 假设 M、N 和 K 是相应大小的数组
    float M[height_M * width_M];
    float N[width_M * width_N];
    float K[height_M * width_N];

    // 初始化数组（这里只是示例，请根据你的实际需求初始化数据）
    for (int i = 0; i < height_M * width_M; i++) {
        M[i] = i + 1;
    }

    for (int i = 0; i < width_M * width_N; i++) {
        N[i] = i + 2;
    }

    // 调用函数
    mul(M, N, K, height_M, width_N, width_M);

    // 在这里可以添加打印结果或其他操作
    // 例如：
    for (int i = 0; i < height_M; i++) {
        for (int j = 0; j < width_N; j++) {
            printf("%f ", K[i * (int)width_N + j]);
        }
        printf("\n");
    }

    return 0;
}

void mul(float *M, float *N, float *K, float height_M, float width_N, float width_M) {
    for (int i = 0; i < height_M; i++) {
        for (int j = 0; j < width_N; j++) {
            float sum = 0;
            for (int k = 0; k < width_M; k++) {
                float a = M[i * (int)width_M + k];
                float b = N[k * (int)width_N + j];
                sum += a * b;
            }
            K[i * (int)width_N + j] = sum;
        }
    }
}
