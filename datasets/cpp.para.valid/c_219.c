#include <stdio.h>

// 函数声明
void saxpi_c(int n, float a, float *x, float *y);

int main() {
    // 示例数据
    const int n = 5;
    float a = 2.0;
    float x[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    float y[] = {5.0, 4.0, 3.0, 2.0, 1.0};

    // 调用函数
    saxpi_c(n, a, x, y);

    // 输出结果
    printf("Resultant vector after saxpi_c operation:\n");
    for (int i = 0; i < n; i++) {
        printf("%f ", y[i]);
    }

    return 0;
}

// 函数定义
void saxpi_c(int n, float a, float *x, float *y) {
    for (int i = 0; i < n; i++) {
        y[i] = a * x[i] + y[i];
    }
}
 
