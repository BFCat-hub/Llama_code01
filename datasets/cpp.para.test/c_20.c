#include <stdio.h>

void initWith_cpu(float num, float *a, int N) {
    for (int i = 0; i < N; i++) {
        a[i] = num;
    }
}

int main() {
    // 示例用法
    int arraySize = 5;
    float arrayA[arraySize];
    float initialValue = 3.14;

    // 调用函数
    initWith_cpu(initialValue, arrayA, arraySize);

    printf("初始化后的数组 A：");
    for (int i = 0; i < arraySize; i++) {
        printf("%.2f ", arrayA[i]);
    }

    return 0;
}
