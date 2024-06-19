#include <stdio.h>

// 函数声明
void incrementArrayOnHost(float *a, int N);

int main() {
    // 示例数据
    int N = 5;
    float a[] = {1.0, 2.0, 3.0, 4.0, 5.0};

    // 调用函数
    incrementArrayOnHost(a, N);

    // 输出结果
    printf("Array after incrementing each element by 1.0:\n");
    for (int i = 0; i < N; i++) {
        printf("%f ", a[i]);
    }

    return 0;
}

// 函数定义
void incrementArrayOnHost(float *a, int N) {
    for (int i = 0; i < N; i++) {
        a[i] = a[i] + 1.0f;
    }
}
 
