#include <stdio.h>

void memsetCpuInt(int *data, int val, int N) {
    for (int index = 0; index < N; index++) {
        data[index] = val;
    }
}

int main() {
    // 示例用法
    int numElements = 5;
    int array[] = {1, 2, 3, 4, 5};
    int value = 42;

    printf("原始数组：");
    for (int i = 0; i < numElements; i++) {
        printf("%d ", array[i]);
    }

    // 调用函数
    memsetCpuInt(array, value, numElements);

    printf("\n设置后的数组：");
    for (int i = 0; i < numElements; i++) {
        printf("%d ", array[i]);
    }

    return 0;
}
