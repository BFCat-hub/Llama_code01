#include <stdio.h>

// 函数声明
void set_offset_kernel(int stride, int size, int *output);

int main() {
    // 示例数据
    int stride = 2;
    int size = 5;
    int output[size];

    // 调用函数
    set_offset_kernel(stride, size, output);

    // 输出结果
    printf("Array after setting offset values:\n");
    for (int i = 0; i < size; i++) {
        printf("%d ", output[i]);
    }

    return 0;
}

// 函数定义
void set_offset_kernel(int stride, int size, int *output) {
    for (int i = 0; i < size; i++) {
        output[i] = i * stride;
    }
}
 
