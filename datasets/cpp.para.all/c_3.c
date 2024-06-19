#include <stdio.h>

void square(int *array, int arrayCount) {
    for (int idx = 0; idx < arrayCount; idx++) {
        array[idx] *= array[idx];
    }
}

int main() {
    // 示例用法
    int array[] = {2, 4, 6, 8, 10};
    int numElements = sizeof(array) / sizeof(array[0]);

    printf("原始数组：");
    for (int i = 0; i < numElements; i++) {
        printf("%d ", array[i]);
    }

    // 调用函数
    square(array, numElements);

    printf("\n平方后的数组：");
    for (int i = 0; i < numElements; i++) {
        printf("%d ", array[i]);
    }

    return 0;
}
