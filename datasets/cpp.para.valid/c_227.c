#include <stdio.h>

// 函数声明
void vadd(const float *a, const float *b, float *c, const unsigned int count);

int main() {
    // 示例数据
    const unsigned int count = 5;
    float a[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    float b[] = {5.0, 4.0, 3.0, 2.0, 1.0};
    float c[5];

    // 调用函数
    vadd(a, b, c, count);

    // 输出结果
    printf("Resultant array after elementwise addition:\n");
    for (unsigned int i = 0; i < count; i++) {
        printf("%f ", c[i]);
    }

    return 0;
}

// 函数定义
void vadd(const float *a, const float *b, float *c, const unsigned int count) {
    for (unsigned int i = 0; i < count; i++) {
        c[i] = a[i] + b[i];
    }
}
 
