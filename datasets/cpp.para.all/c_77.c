#include <stdio.h>

void addMatrix(float *a, float *b, float *c, int N) {
    int i, j, idx;
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            idx = i * N + j;
            a[idx] = b[idx] + c[idx];
        }
    }
}

int main() {
    // 示例用法
    int N = 3;   // 你的 N 值
    float *a = new float[N * N];
    float *b = new float[N * N];
    float *c = new float[N * N];

    // 假设 b 和 c 已经被赋值

    // 调用函数
    addMatrix(a, b, c, N);

    // 打印结果
    printf("矩阵相加后的结果（a矩阵）：\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            int idx = i * N + j;
            printf("%.2f ", a[idx]);
        }
        printf("\n");
    }

    // 释放内存
    delete[] a;
    delete[] b;
    delete[] c;

    return 0;
}
