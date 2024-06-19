#include <stdio.h>
#include <math.h>

void softmax_x_ent_cpu(int n, float *pred, float *truth, float *delta, float *error);

int main() {
    // 在这里可以创建测试用的数据，并调用 softmax_x_ent_cpu 函数
    // 例如：
    int n = 5;  // 你的数组大小

    // 假设 pred、truth、delta 和 error 是相应大小的数组
    float pred[n];
    float truth[n];
    float delta[n];
    float error[n];

    // 初始化数组（这里只是示例，请根据你的实际需求初始化数据）
    for (int i = 0; i < n; i++) {
        pred[i] = 0.2;
        truth[i] = (i == 2) ? 1.0 : 0.0;  // 设置一个位置为 1，表示真实类别
    }

    // 调用函数
    softmax_x_ent_cpu(n, pred, truth, delta, error);

    // 在这里可以添加打印结果或其他操作
    // 例如：
    printf("Error: ");
    for (int i = 0; i < n; i++) {
        printf("%f ", error[i]);
    }
    printf("\n");

    printf("Delta: ");
    for (int i = 0; i < n; i++) {
        printf("%f ", delta[i]);
    }
    printf("\n");

    return 0;
}

void softmax_x_ent_cpu(int n, float *pred, float *truth, float *delta, float *error) {
    int i;
    for (i = 0; i < n; ++i) {
        float t = truth[i];
        float p = pred[i];
        error[i] = (t) ? -log(p) : 0;
        delta[i] = t - p;
    }
}
