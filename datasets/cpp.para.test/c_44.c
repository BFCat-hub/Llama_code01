#include <stdio.h>

void host_add(float *c, float *a, float *b, int n) {
    for (int k = 0; k < n; k++) {
        c[k] = a[k] + b[k];
    }
}

int main() {
    // 示例用法
    int arraySize = 4;
    float arrayA[] = {1.0, 2.0, 3.0, 4.0};
    float arrayB[] = {5.0, 6.0, 7.0, 8.0};
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
    host_add(resultArray, arrayA, arrayB, arraySize);

    printf("\n数组 C（相加后）：");
    for (int i = 0; i < arraySize; i++) {
        printf("%.2f ", resultArray[i]);
    }

    return 0;
}
