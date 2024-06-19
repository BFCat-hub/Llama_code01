#include <stdio.h>
#include <math.h>

void cpuChoiLee(float *xi, float *xq, float *sr, float *si, int N, float *L);

int main() {
    // 示例输入数据，你可以根据实际情况修改
    int N = 10;
    float xi[N], xq[N], sr[N], si[N], L[N];

    // 初始化输入数据，这里仅为示例，请根据实际情况修改
    for (int i = 0; i < N; ++i) {
        xi[i] = i + 1;
        xq[i] = i + 2;
        sr[i] = i + 3;
        si[i] = i + 4;
    }

    // 调用函数
    cpuChoiLee(xi, xq, sr, si, N, L);

    // 打印结果，这里仅为示例，请根据实际情况修改
    printf("Results:\n");
    for (int i = 0; i < N; ++i) {
        printf("L[%d] = %f\n", i, L[i]);
    }

    return 0;
}

void cpuChoiLee(float *xi, float *xq, float *sr, float *si, int N, float *L) {
    for (int u = 0; u < N; u++) {
        float uSum = 0;
        float r_i, r_q, rconj_i, rconj_q;
        float s_i, s_q, sconj_i, sconj_q;
        float rsum_i, rsum_q, ssum_i, ssum_q;
        float ksum_i, ksum_q;

        for (int i = 0; i < N; i++) {
            ksum_i = 0;
            ksum_q = 0;

            for (int k = 0; k < N - i; k++) {
                r_i = xi[u + k + i];
                r_q = xq[u + k + i];
                rconj_i = xi[u + k];
                rconj_q = xq[u + k] * (-1);
                s_i = sr[k];
                s_q = si[k];
                sconj_i = sr[k + i];
                sconj_q = si[k + i] * (-1);
                rsum_i = (r_i * rconj_i) - (r_q * rconj_q);
                rsum_q = (r_i * rconj_q) + (r_q * rconj_i);
                ssum_i = (s_i * sconj_i) - (s_q * sconj_q);
                ssum_q = (s_i * sconj_q) + (s_q * sconj_i);
                ksum_i += (rsum_i * ssum_i) - (rsum_q * ssum_q);
                ksum_q += (rsum_i * ssum_q) + (rsum_q * ssum_i);
            }

            uSum += sqrt((ksum_i * ksum_i) + (ksum_q * ksum_q));
        }

        L[u] = uSum;
    }
}
