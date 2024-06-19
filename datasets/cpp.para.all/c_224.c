#include <stdio.h>

// 函数声明
void vectorAdd(double *a, double *b, double *c, int vector_size);

int main() {
    // 示例数据
    const int vector_size = 5;
    double a[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    double b[] = {5.0, 4.0, 3.0, 2.0, 1.0};
    double c[5];

    // 调用函数
    vectorAdd(a, b, c, vector_size);

    // 输出结果
    printf("Resultant array after elementwise addition:\n");
    for (int i = 0; i < vector_size; i++) {
        printf("%f ", c[i]);
    }

    return 0;
}

// 函数定义
void vectorAdd(double *a, double *b, double *c, int vector_size) {
    for (int idx = 0; idx < vector_size; idx++) {
        c[idx] = a[idx] + b[idx];
    }
}
 
