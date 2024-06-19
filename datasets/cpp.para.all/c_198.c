#include <stdio.h>

// 函数声明
void allLog2_cpu(const double *arr, double *buf, int n);

int main() {
    // 示例数据
    int n = 4;
    double arr[] = {2.0, 4.0, 8.0, 16.0};
    double buf[4];

    // 调用函数
    allLog2_cpu(arr, buf, n);

    // 输出结果
    printf("Array after dividing each element by 2:\n");
    for (int i = 0; i < n; i++) {
        printf("%f ", buf[i]);
    }

    return 0;
}

// 函数定义
void allLog2_cpu(const double *arr, double *buf, int n) {
    for (int i = 0; i < n; i++) {
        buf[i] = arr[i] / 2.0;
    }
}
 
