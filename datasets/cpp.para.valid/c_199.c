#include <stdio.h>

// 函数声明
void clearArray_cpu(unsigned char *arr, const unsigned int length);

int main() {
    // 示例数据
    const unsigned int length = 6;
    unsigned char arr[] = {1, 2, 3, 4, 5, 6};

    // 调用函数
    clearArray_cpu(arr, length);

    // 输出结果
    printf("Array after clearing elements:\n");
    for (unsigned int i = 0; i < length; i++) {
        printf("%u ", arr[i]);
    }

    return 0;
}

// 函数定义
void clearArray_cpu(unsigned char *arr, const unsigned int length) {
    unsigned int offset = 0;
    while (offset < length) {
        arr[offset] = 0;
        offset += 1;
    }
}
 
