#include <stdio.h>

// 函数声明
void saxpy_cpu(float *x, float *y, float alpha, int n);

int main() {
    // 示例数据
    int n = 5;
    float x[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    float y[] = {5.0, 4.0, 3.0, 2.0, 1.0};
    float alpha = 2.0;

    // 调用函数
    saxpy_cpu(x, y, alpha, n);

    // 输出结果
    printf("Resultant vector after saxpy operation:\n");
    for (int i = 0; i < n; i++) {
        printf("%f ", y[i]);
    }

    return 0;
}

// 函数定义
void saxpy_cpu(float *x, float *y, float alpha, int n) {
    for (int i = 0; i < n; i++) {
        y[i] = alpha * x[i] + y[i];
    }
}
 
