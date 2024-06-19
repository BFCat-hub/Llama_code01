#include <stdio.h>

void pathPlan(int *devSpeed, int *devSteer, int size) {
    for (int tid = 0; tid < size; tid++) {
        devSpeed[tid] += 1;
        devSteer[tid] += 1;
    }
}

int main() {
    // 示例用法
    int arraySize = 4;
    int speedArray[] = {10, 20, 30, 40};
    int steerArray[] = {1, 2, 3, 4};

    printf("速度数组：");
    for (int i = 0; i < arraySize; i++) {
        printf("%d ", speedArray[i]);
    }

    printf("\n方向数组：");
    for (int i = 0; i < arraySize; i++) {
        printf("%d ", steerArray[i]);
    }

    // 调用函数
    pathPlan(speedArray, steerArray, arraySize);

    printf("\n计划后的速度数组：");
    for (int i = 0; i < arraySize; i++) {
        printf("%d ", speedArray[i]);
    }

    printf("\n计划后的方向数组：");
    for (int i = 0; i < arraySize; i++) {
        printf("%d ", steerArray[i]);
    }

    return 0;
}
