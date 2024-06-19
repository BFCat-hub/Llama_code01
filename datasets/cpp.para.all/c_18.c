#include <stdio.h>

void host_add(float *c, float *a, float *b, int n) {
    for (int k = 0; k < n; k++) {
        c[k] = a[k] + b[k];
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
    host_add(resultArray, arrayA, arrayB, arraySize);

    printf("\n相加后的数组 C：");
    for (int i = 0; i < arraySize; i++) {
        printf("%.2f ", resultArray[i]);
    }

    return 0;
}
