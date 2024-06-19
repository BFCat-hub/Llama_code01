#include <stdio.h>

// 函数声明
void allDivInplace_cpu(double *arr, double alpha, int n);

int main() {
    // 示例数据
    int n = 6;
    double arr[] = {2.0, 4.0, 6.0, 8.0, 10.0, 12.0};
    double alpha = 2.0;

    // 调用函数
    allDivInplace_cpu(arr, alpha, n);

    // 输出结果
    printf("Array after dividing each element by %f:\n", alpha);
    for (int i = 0; i < n; i++) {
        printf("%f ", arr[i]);
    }

    return 0;
}

// 函数定义
void allDivInplace_cpu(double *arr, double alpha, int n) {
    for (int i = 0; i < n; i++) {
        arr[i] /= alpha;
    }
}
 
