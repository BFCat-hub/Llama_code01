#include <stdio.h>

// 函数声明
void vector_add_cpu(float a[], float b[], float *c);

int main() {
    // 示例数据
    float a[10000], b[10000], c[10000];

    // 初始化数组 a 和 b
    for (int i = 0; i < 10000; i++) {
        a[i] = i;
        b[i] = i * 2;
    }

    // 调用函数
    vector_add_cpu(a, b, c);

    // 输出结果
    printf("Resultant array after vector addition:\n");
    for (int i = 0; i < 10; i++) {
        printf("%f ", c[i]);
    }

    return 0;
}

// 函数定义
void vector_add_cpu(float a[], float b[], float *c) {
    for (int i = 0; i < 10000; i++) {
        c[i] = a[i] + b[i];
    }
}
 
