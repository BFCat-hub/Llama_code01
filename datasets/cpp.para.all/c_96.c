#include <stdio.h>
#include <stdlib.h>

void cpu_sgemm(float *C, float *A, float *B, long size) {
    for (long i = 0; i < size; i++) {
        for (long k = 0; k < size; k++) {
            for (long j = 0; j < size; j++) {
                C[i * size + j] += A[i * size + k] * B[k * size + j];
            }
        }
    }
}

int main() {
    // 设置示例数据大小
    const long size = 3;

    // 分配内存
    float *A = (float *)malloc(size * size * sizeof(float));
    float *B = (float *)malloc(size * size * sizeof(float));
    float *C = (float *)malloc(size * size * sizeof(float));

    // 填充示例输入数据，这里只是一个例子，实际应用中需要根据具体情况进行初始化
    for (long i = 0; i < size * size; i++) {
        A[i] = i + 1;
        B[i] = i - 1;
        C[i] = 0.0;
    }

    // 调用函数进行矩阵乘法
    cpu_sgemm(C, A, B, size);

    // 打印输出结果，这里只是一个例子，实际应用中可以根据需要进行处理
    printf("Resultant matrix C:\n");
    for (long i = 0; i < size; i++) {
        for (long j = 0; j < size; j++) {
            printf("%f ", C[i * size + j]);
        }
        printf("\n");
    }

    // 释放内存
    free(A);
    free(B);
    free(C);

    return 0;
}
