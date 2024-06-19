#include <stdio.h>

void dsubtract_matrix(double *a, double *b, double *c, int N) {
    for (int idx = 0; idx < N; idx++) {
        c[idx] = a[idx] - b[idx];
    }
}

int main() {
    // 示例用法
    int arraySize = 6;
    double arrayA[] = {1.1, 2.2, 3.3, 4.4, 5.5, 6.6};
    double arrayB[] = {0.5, 1.5, 2.5, 3.5, 4.5, 5.5};
    double resultArray[arraySize];

    printf("数组 A：");
    for (int i = 0; i < arraySize; i++) {
        printf("%.2f ", arrayA[i]);
    }

    printf("\n数组 B：");
    for (int i = 0; i < arraySize; i++) {
        printf("%.2f ", arrayB[i]);
    }

    // 调用函数
    dsubtract_matrix(arrayA, arrayB, resultArray, arraySize);

    printf("\n数组 C（差）：");
    for (int i = 0; i < arraySize; i++) {
        printf("%.2f ", resultArray[i]);
    }

    return 0;
}
