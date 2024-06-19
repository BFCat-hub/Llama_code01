#include <stdio.h>

void sum_arrays_cpu(int *a, int *b, int *c, int size) {
    for (int i = 0; i < size; i++) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    // 示例用法
    int arraySize = 5;
    int arrayA[] = {1, 2, 3, 4, 5};
    int arrayB[] = {10, 20, 30, 40, 50};
    int resultArray[arraySize];

    printf("数组 A：");
    for (int i = 0; i < arraySize; i++) {
        printf("%d ", arrayA[i]);
    }

    printf("\n数组 B：");
    for (int i = 0; i < arraySize; i++) {
        printf("%d ", arrayB[i]);
    }

    // 调用函数
    sum_arrays_cpu(arrayA, arrayB, resultArray, arraySize);

    printf("\n数组 C（和）：");
    for (int i = 0; i < arraySize; i++) {
        printf("%d ", resultArray[i]);
    }

    return 0;
}
