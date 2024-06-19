#include <stdio.h>
#include <stdlib.h>

void copyAliasRow(int *devMat, int memWidth, int memHeight, int size) {
    for (int devMatX = 0; devMatX < size; devMatX++) {
        devMat[memWidth * 0 + devMatX] = devMat[memWidth * (memHeight - 2) + devMatX];
        devMat[memWidth * (memHeight - 1) + devMatX] = devMat[memWidth * 1 + devMatX];
    }
}

int main() {
    // 设置示例数据大小
    int memWidth = 4;
    int memHeight = 4;
    int size = memWidth;

    // 分配内存
    int *devMat = (int *)malloc(memWidth * memHeight * sizeof(int));

    // 填充示例输入数据，这里只是一个例子，实际应用中需要根据具体情况进行初始化
    for (int i = 0; i < memWidth * memHeight; i++) {
        devMat[i] = i + 1;
    }

    // 调用函数进行行复制
    copyAliasRow(devMat, memWidth, memHeight, size);

    // 打印输出结果，这里只是一个例子，实际应用中可以根据需要进行处理
    printf("devMat after copyAliasRow:\n");
    for (int i = 0; i < memHeight; i++) {
        for (int j = 0; j < memWidth; j++) {
            printf("%d ", devMat[i * memWidth + j]);
        }
        printf("\n");
    }

    // 释放内存
    free(devMat);

    return 0;
}
