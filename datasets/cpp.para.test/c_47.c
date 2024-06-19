#include <stdio.h>

void fill_matrix(double *const A, const int rows, const int cols) {
    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
            A[row * cols + col] = row;
        }
    }
}

int main() {
    // 示例用法
    int numRows = 3;
    int numCols = 4;
    double matrix[numRows * numCols];

    // 调用函数
    fill_matrix(matrix, numRows, numCols);

    // 打印填充后的矩阵
    printf("填充后的矩阵：\n");
    for (int row = 0; row < numRows; row++) {
        for (int col = 0; col < numCols; col++) {
            printf("%.2f ", matrix[row * numCols + col]);
        }
        printf("\n");
    }

    return 0;
}
