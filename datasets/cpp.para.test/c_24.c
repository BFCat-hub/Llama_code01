#include <stdio.h>

void sum_array_cpu(float *a, float *b, float *c, const int size) {
    for (int i = 0; i < size; ++i) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    // 示例用法
    int arraySize = 5;
    float arrayA[] = {1.1, 2.2, 3.3, 4.4, 5.5};
    float arrayB[] = {0.5, 1.5, 2.5, 3.5, 4.5};
    float resultArray[arraySize];

    printf("数组 A：");
    for (int i = 0; i < arraySize; i++) {
        printf("%.2f ", arrayA[i]);
    }

    printf("\n数组 B：");
    for (int i = 0; i < arraySize; i++) {
        printf("%.2f ", arrayB[i]);
    }

    // 调用函数
    sum_array_cpu(arrayA, arrayB, resultArray, arraySize);

    printf("\n数组 C（和）：");
    for (int i = 0; i < arraySize; i++) {
        printf("%.2f ", resultArray[i]);
    }

    return 0;
}
