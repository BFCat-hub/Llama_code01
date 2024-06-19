#include <stdio.h>
#include <stdlib.h>

int cpuReduce(int *N, const int size) {
    if (size == 1)
        return N[0];

    int stride = size / 2;

    for (int i = 0; i < stride; i++)
        N[i] += N[i + stride];

    return cpuReduce(N, stride);
}

int main() {
    // 设置示例数据大小
    const int size = 8;

    // 分配内存
    int *N = (int *)malloc(size * sizeof(int));

    // 填充示例输入数据，这里只是一个例子，实际应用中需要根据具体情况进行初始化
    for (int i = 0; i < size; i++) {
        N[i] = i + 1;
    }

    // 调用函数进行 Reduce
    int result = cpuReduce(N, size);

    // 打印输出结果，这里只是一个例子，实际应用中可以根据需要进行处理
    printf("Reduced value: %d\n", result);

    // 释放内存
    free(N);

    return 0;
}
