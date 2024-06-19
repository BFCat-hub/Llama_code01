#include <stdio.h>

// 函数声明
void vectorAdd(const float *A, const float *B, float *C, int numElements);

int main() {
    // 示例数据
    const int numElements = 5;
    float A[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    float B[] = {5.0, 4.0, 3.0, 2.0, 1.0};
    float C[5];

    // 调用函数
    vectorAdd(A, B, C, numElements);

    // 输出结果
    printf("Resultant array after elementwise addition:\n");
    for (int i = 0; i < numElements; i++) {
        printf("%f ", C[i]);
    }

    return 0;
}

// 函数定义
void vectorAdd(const float *A, const float *B, float *C, int numElements) {
    for (int i = 0; i < numElements; i++) {
        C[i] = A[i] + B[i];
    }
}
 
