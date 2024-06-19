#include <stdio.h>
#include <math.h>

void k_adam_kernel(float *m, float *v, float *w, const float *d, int max_size, float beta1, float beta2, float beta1_tpower, float beta2_tpower, float learning_rate);

int main() {
    // 在这里可以创建测试用的数据，并调用 k_adam_kernel 函数
    // 例如：
    int max_size = 5;
    float beta1 = 0.9;
    float beta2 = 0.999;
    float beta1_tpower = 1.0;
    float beta2_tpower = 1.0;
    float learning_rate = 0.001;

    // 假设 m, v, w, 和 d 是相应大小的数组
    float m[max_size];
    float v[max_size];
    float w[max_size];
    float d[max_size];

    // 初始化数组（这里只是示例，请根据你的实际需求初始化数据）
    for (int i = 0; i < max_size; i++) {
        m[i] = i + 1;
        v[i] = i + 2;
        w[i] = i + 3;
        d[i] = i + 4;
    }

    // 调用函数
    k_adam_kernel(m, v, w, d, max_size, beta1, beta2, beta1_tpower, beta2_tpower, learning_rate);

    // 在这里可以添加打印结果或其他操作
    // 例如：
    printf("Updated w: ");
    for (int i = 0; i < max_size; i++) {
        printf("%f ", w[i]);
    }
    printf("\n");

    return 0;
}

void k_adam_kernel(float *m, float *v, float *w, const float *d, int max_size, float beta1, float beta2, float beta1_tpower, float beta2_tpower, float learning_rate) {
    const float eps = 1e-8;

    for (int i = 0; i < max_size; i++) {
        float d_temp = d[i];
        m[i] = m[i] * beta1 + d_temp * (1 - beta1);
        v[i] = v[i] * beta2 + d_temp * d_temp * (1 - beta2);

        float m_hat = m[i] / (1 - beta1_tpower);
        float v_hat = sqrt(v[i] / (1 - beta2_tpower)) + eps;

        w[i] += (m_hat / v_hat) * (-learning_rate);
    }
}
