#include <stdio.h>

void InitCCL(int labelList[], int reference[], int width, int height) {
    int x, y;
    for (x = 0; x < width; x++) {
        for (y = 0; y < height; y++) {
            int id = x + y * width;
            labelList[id] = reference[id] = id;
        }
    }
}

int main() {
    // 示例用法
    int width = 3;
    int height = 3;

    // 分配内存并初始化数组
    int *labelList = new int[width * height];
    int *reference = new int[width * height];

    // 调用函数
    InitCCL(labelList, reference, width, height);

    // 打印结果
    printf("初始化后的 labelList 数组：\n");
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int id = x + y * width;
            printf("%d ", labelList[id]);
        }
        printf("\n");
    }

    printf("\n初始化后的 reference 数组：\n");
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int id = x + y * width;
            printf("%d ", reference[id]);
        }
        printf("\n");
    }

    // 释放内存
    delete[] labelList;
    delete[] reference;

    return 0;
}
