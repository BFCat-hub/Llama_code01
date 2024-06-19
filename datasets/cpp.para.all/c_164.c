#include <stdio.h>

void cpuBYUSimplified(float *xi, float *xq, float *sr, float *si, int N, int Lq, float *L);

int main() {
    // 在这里可以创建测试用的数据，并调用 cpuBYUSimplified 函数
    // 例如：
    int N = 10;
    int Lq = 5;

    // 假设 xi, xq, sr, si, 和 L 是相应大小的数组
    float xi[N * 8 * Lq];
    float xq[N * 8 * Lq];
    float sr[Lq];
    float si[Lq];
    float L[N];

    // 初始化数组（这里只是示例，请根据你的实际需求初始化数据）
    for (int i = 0; i < N * 8 * Lq; i++) {
        xi[i] = i + 1;
        xq[i] = i + 2;
    }

    for (int i = 0; i < Lq; i++) {
        sr[i] = i + 3;
        si[i] = i + 4;
    }

    // 调用函数
    cpuBYUSimplified(xi, xq, sr, si, N, Lq, L);

    // 在这里可以添加打印结果或其他操作
    // 例如：
    for (int i = 0; i < N; i++) {
        printf("%f ", L[i]);
    }
    printf("\n");

    return 0;
}

void cpuBYUSimplified(float *xi, float *xq, float *sr, float *si, int N, int Lq, float *L) {
    for (int u = 0; u < N; u++) {
        float uSum = 0;
        float r_i, r_q, q_i, q_q;
        float realPart, imagPart;

        for (int k = 0; k <= 7; k++) {
            realPart = 0;
            imagPart = 0;

            for (int l = 0; l < Lq; l++) {
                r_i = xi[u + k * Lq + l];
                r_q = xq[u + k * Lq + l];
                q_i = sr[l];
                q_q = si[l] * (-1);

                realPart += (r_i * q_i) - (r_q * q_q);
                imagPart += (r_i * q_q) + (r_q * q_i);
            }

            uSum += (realPart * realPart) + (imagPart * imagPart);
        }

        L[u] = uSum;
    }
}
