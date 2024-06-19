#include <stdio.h>

void InitReduction(int *flags, int voxelCount, int *reduction, int reductionSize) {
    for (int tid = 0; tid < reductionSize; tid++) {
        reduction[tid] = (tid < voxelCount) ? flags[tid] : 0;
    }
}

int main() {
    // 示例用法
    int voxelCount = 4;
    int reductionSize = 6;
    int flagsArray[] = {1, 0, 1, 0};
    int reductionArray[reductionSize];

    printf("标志数组：");
    for (int i = 0; i < voxelCount; i++) {
        printf("%d ", flagsArray[i]);
    }

    // 调用函数
    InitReduction(flagsArray, voxelCount, reductionArray, reductionSize);

    printf("\n初始化的缩减数组：");
    for (int i = 0; i < reductionSize; i++) {
        printf("%d ", reductionArray[i]);
    }

    return 0;
}
