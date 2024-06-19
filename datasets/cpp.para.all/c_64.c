#include <stdio.h>

void matVecColAddInplace_cpu(double *mat, const double *vec, int m, int n) {
    for (int index = 0; index < m * n; index++) {
        int i = index / n;
        int j = index % n;
        mat[i * n + j] += vec[i];
    }
}

int main() {
    // 示例用法
    int rows = 3;
    int cols = 4;
    double matrix[] = {1.0, 2.0, 3.0, 4.0,
                       5.0, 6.0, 7.0, 8.0,
                       9.0, 10.0, 11.0, 12.0};

    double vector[] = {0.5, 1.0, 1.5};

    printf("输入矩阵：\n");
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%.2f ", matrix[i * cols + j]);
        }
        printf("\n");
    }

    // 调用函数
    matVecColAddInplace_cpu(matrix, vector, rows, cols);

    printf("\n每列加上向量后的矩阵：\n");
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%.2f ", matrix[i * cols + j]);
        }
        printf("\n");
    }

    return 0;
}
