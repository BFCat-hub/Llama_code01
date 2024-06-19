#include <stdio.h>

void multiplyIntValues(int *destination, int *vector, int value, unsigned int end) {
    for (unsigned int i = 0; i < end; i++) {
        destination[i] = vector[i] * value;
    }
}

int main() {
    // 示例用法
    unsigned int arraySize = 5;
    int vector[] = {1, 2, 3, 4, 5};
    int resultArray[arraySize];
    int multiplier = 3;

    printf("原始向量：");
    for (unsigned int i = 0; i < arraySize; i++) {
        printf("%d ", vector[i]);
    }

    // 调用函数
    multiplyIntValues(resultArray, vector, multiplier, arraySize);

    printf("\n乘法后的数组：");
    for (unsigned int i = 0; i < arraySize; i++) {
        printf("%d ", resultArray[i]);
    }

    return 0;
}
