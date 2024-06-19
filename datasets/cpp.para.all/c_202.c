#include <stdio.h>

// 函数声明
void host_add(float *c, float *a, float *b, int n);

int main() {
    // 示例数据
    int n = 4;
    float a[] = {1.0, 2.0, 3.0, 4.0};
    float b[] = {5.0, 6.0, 7.0, 8.0};
    float c[4];

    // 调用函数
    host_add(c, a, b, n);

    // 输出结果
    printf("Array c after adding elements from a and b:\n");
    for (int i = 0; i < n; i++) {
        printf("%f ", c[i]);
    }

    return 0;
}

// 函数定义
void host_add(float *c, float *a, float *b, int n) {
    for (int k = 0; k < n; k++) {
        c[k] = a[k] + b[k];
    }
}
 
