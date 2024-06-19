#include <stdio.h>

// 函数声明
void sumArraysOnHostx(int *A, int *B, int *C, const int N);

int main() {
    // 示例数据
    const int N = 5;
    int A[] = {1, 2, 3, 4, 5};
    int B[] = {5, 4, 3, 2, 1};
    int C[5];

    // 调用函数
    sumArraysOnHostx(A, B, C, N);

    // 输出结果
    printf("Resultant array after sumArraysOnHostx:\n");
    for (int i = 0; i < N; i++) {
        printf("%d ", C[i]);
    }

    return 0;
}

// 函数定义
void sumArraysOnHostx(int *A, int *B, int *C, const int N) {
    for (int idx = 0; idx < N; idx++) {
        C[idx] = A[idx] + B[idx];
    }
}
 
