#include <stdio.h>

void dmul_Scalar_matrix(double *a, double value, double *c, int N) {
    for (int idx = 0; idx < N; idx++) {
        c[idx] = a[idx] * value;
    }
}

int main() {
    // 示例用法
    int arraySize = 6;
    double arrayA[] = {1.1, 2.2, 3.3, 4.4, 5.5, 6.6};
    double scalarValue = 2.0;
    double resultArray[arraySize];

    printf("数组 A：");
    for (int i = 0; i < arraySize; i++) {
        printf("%.2f ", arrayA[i]);
    }

    // 调用函数
    dmul_Scalar_matrix(arrayA, scalarValue, resultArray, arraySize);

    printf("\n标量乘法后的数组 C：");
    for (int i = 0; i < arraySize; i++) {
        printf("%.2f ", resultArray[i]);
    }

    return 0;
}
