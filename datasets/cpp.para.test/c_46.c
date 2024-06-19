#include <stdio.h>

void doubleArrayVectorAdd_cpu(double *d_in_a, double *d_in_b, double *d_out, int length) {
    for (int idx = 0; idx < length; idx++) {
        d_out[idx] = d_in_a[idx] + d_in_b[idx];
    }
}

int main() {
    // 示例用法
    int arraySize = 3;
    double arrayA[] = {1.5, 2.5, 3.5};
    double arrayB[] = {0.5, 1.0, 1.5};
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
    doubleArrayVectorAdd_cpu(arrayA, arrayB, resultArray, arraySize);

    printf("\n数组 C（相加后）：");
    for (int i = 0; i < arraySize; i++) {
        printf("%.2f ", resultArray[i]);
    }

    return 0;
}
