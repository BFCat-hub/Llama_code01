#include <stdio.h>

void matColMeanDiv_cpu(double *buf, int m, int n, double *tmp) {
    for (int i = 0; i < n; i++) {
        buf[i] = tmp[i] / m;
    }
}

int main() {
    // 示例用法
    int numRows = 3;
    int numCols = 4;
    double matrix[numRows][numCols] = {{1.0, 2.0, 3.0, 4.0},
                                       {5.0, 6.0, 7.0, 8.0},
                                       {9.0, 10.0, 11.0, 12.0}};
    double resultArray[numCols];

    // 计算每列的均值
    double colMeanArray[numCols] = {0.0};
    for (int i = 0; i < numRows; i++) {
        for (int j = 0; j < numCols; j++) {
            colMeanArray[j] += matrix[i][j];
        }
    }
    for (int j = 0; j < numCols; j++) {
        colMeanArray[j] /= numRows;
    }

    printf("原始矩阵：\n");
    for (int i = 0; i < numRows; i++) {
        for (int j = 0; j < numCols; j++) {
            printf("%.2f ", matrix[i][j]);
        }
        printf("\n");
    }

    // 调用函数
    matColMeanDiv_cpu(resultArray, numRows, numCols, colMeanArray);

    printf("\n每列均值除法后的数组：\n");
    for (int i = 0; i < numCols; i++) {
        printf("%.2f ", resultArray[i]);
    }

    return 0;
}
