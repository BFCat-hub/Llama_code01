#include <stdio.h>

void countRangesGlobal(int size, int *A, int *B) {
    for (int i = 0; i < size; i++) {
        int x = A[i] / 100;
        B[x] += 1;
    }
}

int main() {
    // 示例用法
    int arraySize = 8;
    int inputArray[] = {50, 120, 250, 350, 420, 550, 670, 800};
    int resultArray[9] = {0}; // Assuming ranges from 0 to 800, divided by 100

    printf("原始数组：");
    for (int i = 0; i < arraySize; i++) {
        printf("%d ", inputArray[i]);
    }

    // 调用函数
    countRangesGlobal(arraySize, inputArray, resultArray);

    printf("\n统计后的数组 B：");
    for (int i = 0; i < 9; i++) {
        printf("%d ", resultArray[i]);
    }

    return 0;
}
