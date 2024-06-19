#include <stdio.h>

void transpositionCPU(int *vector, int *transposed, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            transposed[i + j * size] = vector[j + i * size];
        }
    }
}

int main() {
    // 示例用法
    int matrixSize = 3;
    int inputMatrix[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    int transposedMatrix[matrixSize * matrixSize];

    printf("输入矩阵：\n");
    for (int i = 0; i < matrixSize; i++) {
        for (int j = 0; j < matrixSize; j++) {
            printf("%d ", inputMatrix[i * matrixSize + j]);
        }
        printf("\n");
    }

    // 调用函数
    transpositionCPU(inputMatrix, transposedMatrix, matrixSize);

    printf("\n转置后的矩阵：\n");
    for (int i = 0; i < matrixSize; i++) {
        for (int j = 0; j < matrixSize; j++) {
            printf("%d ", transposedMatrix[i * matrixSize + j]);
        }
        printf("\n");
    }

    return 0;
}
