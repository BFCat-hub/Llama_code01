#include <stdio.h>

void allAddInplace_cpu(double *arr, double alpha, int n) {
    for (int i = 0; i < n; i++) {
        arr[i] += alpha;
    }
}

int main() {
    // 示例用法
    int numElements = 5;
    double array[] = {1.1, 2.2, 3.3, 4.4, 5.5};
    double alpha = 10.0;

    printf("原始数组：");
    for (int i = 0; i < numElements; i++) {
        printf("%.2f ", array[i]);
    }

    // 调用函数
    allAddInplace_cpu(array, alpha, numElements);

    printf("\n所有元素加上常数后的数组：");
    for (int i = 0; i < numElements; i++) {
        printf("%.2f ", array[i]);
    }

    return 0;
}
