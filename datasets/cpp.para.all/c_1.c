#include <stdio.h>

void add_100(int numElements, int *data) {
    for (int idx = 0; idx < numElements; idx++) {
        data[idx] += 100;
    }
}

int main() {
    // 示例用法
    int array[] = {1, 2, 3, 4, 5};
    int numElements = sizeof(array) / sizeof(array[0]);

    printf("原始数组：");
    for (int i = 0; i < numElements; i++) {
        printf("%d ", array[i]);
    }

    // 调用函数
    add_100(numElements, array);

    printf("\n修改后的数组：");
    for (int i = 0; i < numElements; i++) {
        printf("%d ", array[i]);
    }

    return 0;
}
