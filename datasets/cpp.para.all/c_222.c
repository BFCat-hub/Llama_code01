#include <stdio.h>

// 函数声明
void cpuSAXPY(int len, float a, float *x, float *y);

int main() {
    // 示例数据
    const int len = 5;
    float a = 2.0;
    float x[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    float y[] = {5.0, 4.0, 3.0, 2.0, 1.0};

    // 调用函数
    cpuSAXPY(len, a, x, y);

    // 输出结果
    printf("Resultant vector after cpuSAXPY operation:\n");
    for (int i = 0; i < len; i++) {
        printf("%f ", y[i]);
    }

    return 0;
}

// 函数定义
void cpuSAXPY(int len, float a, float *x, float *y) {
    for (int i = 0; i < len; i++) {
        y[i] = x[i] * a + y[i];
    }
}
 
