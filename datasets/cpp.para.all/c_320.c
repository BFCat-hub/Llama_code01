#include <stdio.h>

// 声明矩阵乘法函数
void mat_mul_seq(int *m_A, int *m_B, int *m_C, int A_rows, int A_cols, int B_rows, int B_cols);

int main() {
    // 定义两个矩阵 A 和 B
    int A_rows = 2, A_cols = 3;
    int B_rows = 3, B_cols = 4;

    int matrix_A[] = {1, 2, 3, 4, 5, 6};
    int matrix_B[] = {7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18};

    // 定义结果矩阵 C
    int matrix_C[A_rows * B_cols];

    // 调用矩阵乘法函数
    mat_mul_seq(matrix_A, matrix_B, matrix_C, A_rows, A_cols, B_rows, B_cols);

    // 打印结果
    printf("Resulting Matrix C:\n");
    for (int i = 0; i < A_rows; i++) {
        for (int j = 0; j < B_cols; j++) {
            printf("%d ", matrix_C[i * B_cols + j]);
        }
        printf("\n");
    }

    return 0;
}

// 定义矩阵乘法函数
void mat_mul_seq(int *m_A, int *m_B, int *m_C, int A_rows, int A_cols, int B_rows, int B_cols) {
    int sum;
    for (int i = 0; i < A_rows; i++) {
        for (int j = 0; j < B_cols; j++) {
            sum = 0;
            for (int k = 0; k < A_cols; k++) {
                sum += m_A[i * A_cols + k] * m_B[k * B_cols + j];
            }
            m_C[i * B_cols + j] = sum;
        }
    }
}

