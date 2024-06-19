#include <stdio.h>
#include <stdlib.h>

void bubbleSort(int *p, const int size) {
    for (int i = 0; i < size - 1; i++) {
        for (int j = 0; j < size - i - 1; j++) {
            if (p[j] > p[j + 1]) {
                int temp = p[j];
                p[j] = p[j + 1];
                p[j + 1] = temp;
            }
        }
    }
}

int main() {
    // 设置示例数据大小
    const int size = 5;

    // 分配内存
    int *p = (int *)malloc(size * sizeof(int));

    // 填充示例输入数据，这里只是一个例子，实际应用中需要根据具体情况进行初始化
    p[0] = 5;
    p[1] = 3;
    p[2] = 1;
    p[3] = 4;
    p[4] = 2;

    // 调用函数进行冒泡排序
    bubbleSort(p, size);

    // 打印输出结果，这里只是一个例子，实际应用中可以根据需要进行处理
    printf("Sorted array: ");
    for (int i = 0; i < size; i++) {
        printf("%d ", p[i]);
    }

    // 释放内存
    free(p);

    return 0;
}
