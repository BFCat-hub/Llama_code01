#include <stdio.h>

// 函数声明
void vectorDiv(const float *A, const float *B, float *C, int numElements);

int main() {
    // 示例数据
    const int numElements = 5;
    float A[] = {10.0, 20.0, 30.0, 40.0, 50.0};
    float B[] = {2.0, 4.0, 5.0, 8.0, 10.0};
    float C[5];

    // 调用函数
    vectorDiv(A, B, C, numElements);

    // 输出结果
    printf("Resultant array after elementwise division:\n");
    for (int i = 0; i < numElements; i++) {
        printf("%f ", C[i]);
    }

    return 0;
}

// 函数定义
void vectorDiv(const float *A, const float *B, float *C, int numElements) {
    for (int i = 0; i < numElements; i++) {
        C[i] = A[i] / B[i];
    }
}
 
