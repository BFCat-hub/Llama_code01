#include <stdio.h>

// 函数声明
void setLabels_cpu(int *output, int dims, int clsNum);

int main() {
    // 示例数据
    int dims = 10;
    int clsNum = 3;
    int output[10];

    // 调用函数
    setLabels_cpu(output, dims, clsNum);

    // 输出结果
    printf("Array after setting labels:\n");
    for (int i = 0; i < dims; i++) {
        printf("%d ", output[i]);
    }

    return 0;
}

// 函数定义
void setLabels_cpu(int *output, int dims, int clsNum) {
    for (int tid = 0; tid < dims; tid++) {
        output[tid] = tid % clsNum;
    }
}
 
