#include <stdio.h>

// 函数声明
void setSuppressed_cpu(int *suppressed, int dims);

int main() {
    // 示例数据
    int dims = 4;
    int suppressed[dims];

    // 调用函数
    setSuppressed_cpu(suppressed, dims);

    // 输出结果
    printf("Array after setting all elements to zero:\n");
    for (int i = 0; i < dims; i++) {
        printf("%d ", suppressed[i]);
    }

    return 0;
}

// 函数定义
void setSuppressed_cpu(int *suppressed, int dims) {
    for (int tid = 0; tid < dims; tid++) {
        suppressed[tid] = 0;
    }
}
 
