#include <stdio.h>

void add(const int x, const int y, const int WIDTH, int *c, const int *a, const int *b) {
    int i = y * WIDTH + x;
    c[i] = a[i] + b[i];
}

int main() {
    // 示例用法
    const int WIDTH = 3;
    const int HEIGHT = 2;
    const int arraySize = WIDTH * HEIGHT;

    int arrayA[arraySize] = {1, 2, 3, 4, 5, 6};
    int arrayB[arraySize] = {7, 8, 9, 10, 11, 12};
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
    add(1, 1, WIDTH, resultArray, arrayA, arrayB);

    printf("\n数组 C：");
    for (int i = 0; i < arraySize; i++) {
        printf("%d ", resultArray[i]);
    }

    return 0;
}
