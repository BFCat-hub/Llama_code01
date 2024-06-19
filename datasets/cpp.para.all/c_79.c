 
#include <stdio.h>
#include <math.h>

void l1_cpu(int n, float *pred, float *truth, float *delta, float *error) {
    for (int i = 0; i < n; ++i) {
        float diff = truth[i] - pred[i];
        error[i] = fabs(diff);
        delta[i] = diff > 0 ? 1 : -1;
    }
}

int main() {
    // 示例用法
    int n = 5;   // 你的 n 值
    float *pred = new float[n];
    float *truth = new float[n];
    float *delta = new float[n];
    float *error = new float[n];

    // 假设 pred 和 truth 已经被赋值

    // 调用函数
    l1_cpu(n, pred, truth, delta, error);

    // 打印结果
    printf("处理后的 delta 数组：\n");
    for (int i = 0; i < n; i++) {
        printf("%.2f ", delta[i]);
    }

    printf("\n处理后的 error 数组：\n");
    for (int i = 0; i < n; i++) {
        printf("%.2f ", error[i]);
    }

    // 释放内存
    delete[] pred;
    delete[] truth;
    delete[] delta;
    delete[] error;

    return 0;
}
