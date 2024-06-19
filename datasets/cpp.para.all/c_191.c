#include <stdio.h>

// 函数声明
void allExp2Inplace_cpu(double *arr, int n);

int main() {
    // 示例数据
    int n = 4;
    double arr[] = {1.0, 2.0, 3.0, 4.0};

    // 调用函数
    allExp2Inplace_cpu(arr, n);

    // 输出结果
    printf("Array after multiplying each element by 9:\n");
    for (int i = 0; i < n; i++) {
        printf("%f ", arr[i]);
    }

    return 0;
}

// 函数定义
void allExp2Inplace_cpu(double *arr, int n) {
    for (int i = 0; i < n; i++) {
        arr[i] = arr[i] * 9.0;
    }
}
 
