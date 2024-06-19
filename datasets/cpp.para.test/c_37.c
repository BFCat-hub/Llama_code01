#include <stdio.h>

void add_kernel(float *inputleft, float *inputright, float *output, int count) {
    for (int idx = 0; idx < count; idx++) {
        output[idx] = inputleft[idx] + inputright[idx];
    }
}

int main() {
    // 示例用法
    int arraySize = 4;
    float inputLeft[] = {1.1, 2.2, 3.3, 4.4};
    float inputRight[] = {0.5, 1.5, 2.5, 3.5};
    float resultArray[arraySize];

    printf("左输入数组：");
    for (int i = 0; i < arraySize; i++) {
        printf("%.2f ", inputLeft[i]);
    }

    printf("\n右输入数组：");
    for (int i = 0; i < arraySize; i++) {
        printf("%.2f ", inputRight[i]);
    }

    // 调用函数
    add_kernel(inputLeft, inputRight, resultArray, arraySize);

    printf("\n输出数组：");
    for (int i = 0; i < arraySize; i++) {
        printf("%.2f ", resultArray[i]);
    }

    return 0;
}
