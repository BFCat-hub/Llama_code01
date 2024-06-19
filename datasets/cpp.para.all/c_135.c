#include <stdio.h>

void matrMult(float *A, float *B, float *C, int rowsA, int colsA, int colsB) {
    for (int i = 0; i < rowsA; ++i) {
        for (int j = 0; j < colsB; ++j) {
            for (int k = 0; k < colsA; ++k) {
                C[i * colsB + j] += A[i * colsA + k] * B[k * colsB + j];
            }
        }
    }
}

int main() {
    // 示例数据
    const int rowsA = 2;
    const int colsA = 3;
    const int colsB = 2;

    float A[rowsA * colsA] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    float B[colsA * colsB] = {7.0, 8.0, 9.0, 10.0, 11.0, 12.0};
    float C[rowsA * colsB] = {0.0};

    // 调用 matrMult 函数
    matrMult(A, B, C, rowsA, colsA, colsB);

    // 打印输出结果，这里只是一个例子，实际应用中可以根据需要进行处理
    for (int i = 0; i < rowsA; ++i) {
        for (int j = 0; j < colsB; ++j) {
            printf("Resultant C[%d][%d]: %f\n", i, j, C[i * colsB + j]);
        }
    }

    return 0;
}
