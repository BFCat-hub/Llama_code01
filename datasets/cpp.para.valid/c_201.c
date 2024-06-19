#include <stdio.h>

// 函数声明
void add(int n, float *x, float *y);

int main() {
    // 示例数据
    int n = 3;
    float x[] = {1.0, 2.0, 3.0};
    float y[] = {4.0, 5.0, 6.0};

    // 调用函数
    add(n, x, y);

    // 输出结果
    printf("Array y after adding elements from x:\n");
    for (int i = 0; i < n; i++) {
        printf("%f ", y[i]);
    }

    return 0;
}

// 函数定义
void add(int n, float *x, float *y) {
    for (int i = 0; i < n; i++) {
        y[i] = x[i] + y[i];
    }
}
 
