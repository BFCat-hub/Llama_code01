#include <stdio.h>

void AddMatrixOnCPU(int *A, int *B, int *C, int nx, int ny) {
    int i, j;
    int cnt = 0;
    for (j = 0; j < ny; j++) {
        for (i = 0; i < nx; i++) {
            C[cnt] = A[cnt] + B[cnt];
            cnt++;
        }
    }
}

int main() {
    // 示例用法
    int nx = 3;   // 矩阵的列数
    int ny = 3;   // 矩阵的行数
    int size = nx * ny;
    int *A = new int[size];
    int *B = new int[size];
    int *C = new int[size];

    // 假设 A 和 B 矩阵已经被赋值

    // 调用函数
    AddMatrixOnCPU(A, B, C, nx, ny);

    // 打印结果
    printf("处理后的 C 矩阵：\n");
    for (int i = 0; i < size; i++) {
        printf("%d ", C[i]);
    }

    // 释放内存
    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}
