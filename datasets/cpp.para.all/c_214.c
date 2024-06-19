#include <stdio.h>

// 函数声明
void sumArrays_cpu(float *A, float *B, float *C, const int N);

int main() {
    // 示例数据
    const int N = 5;
    float A[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    float B[] = {5.0, 4.0, 3.0, 2.0, 1.0};
    float C[5];

    // 调用函数
    sumArrays_cpu(A, B, C, N);

    // 输出结果
    printf("Resultant array after sumArrays_cpu:\n");
    for (int i = 0; i < N; i++) {
        printf("%f ", C[i]);
    }

    return 0;
}

// 函数定义
void sumArrays_cpu(float *A, float *B, float *C, const int N) {
    for (int i = 0; i < N; i++) {
        C[i] = A[i] + B[i];
    }
}
 
