#include <stdio.h>

// 函数声明
void subAvg_cpu(int *input, int count, int avg);

int main() {
    // 示例数据
    int count = 5;
    int input[] = {2, 4, 6, 8, 10};
    int avg = 6;

    // 调用函数
    subAvg_cpu(input, count, avg);

    // 输出结果
    printf("Array after subtracting average value %d:\n", avg);
    for (int i = 0; i < count; i++) {
        printf("%d ", input[i]);
    }

    return 0;
}

// 函数定义
void subAvg_cpu(int *input, int count, int avg) {
    for (int index = 0; index < count; index++) {
        input[index] = input[index] - avg;
    }
}
 
