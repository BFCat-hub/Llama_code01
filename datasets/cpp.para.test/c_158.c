#include <stdio.h>

void convLayer_forward(int N, int M, int C, int H, int W, int K, float *X, float *Wk, float *Y);

int main() {
    // 在这里可以创建测试用的数据，并调用 convLayer_forward 函数
    // 例如：
    int N = 1;
    int M = 1;
    int C = 1;
    int H = 5;
    int W = 5;
    int K = 3;

    // 假设 X, Wk, 和 Y 是相应大小的数组
    float X[N * C * H * W];
    float Wk[M * C * K * K];
    float Y[N * M * (H - K + 1) * (W - K + 1)];

    // 初始化数组（这里只是示例，请根据你的实际需求初始化数据）
    for (int i = 0; i < N * C * H * W; i++) {
        X[i] = i + 1;
    }

    for (int i = 0; i < M * C * K * K; i++) {
        Wk[i] = i + 2;
    }

    // 调用函数
    convLayer_forward(N, M, C, H, W, K, X, Wk, Y);

    // 在这里可以添加打印结果或其他操作
    // 例如：
    for (int i = 0; i < N * M * (H - K + 1) * (W - K + 1); i++) {
        printf("%f ", Y[i]);
    }
    printf("\n");

    return 0;
}

void convLayer_forward(int N, int M, int C, int H, int W, int K, float *X, float *Wk, float *Y) {
    int n, m, c, h, w, p, q;
    int H_out = H - K + 1;
    int W_out = W - K + 1;

    for (n = 0; n < N; n++)
        for (m = 0; m < M; m++)
            for (h = 0; h < H_out; h++)
                for (w = 0; w < W_out; w++) {
                    Y[n * M * H_out * W_out + m * H_out * W_out + h * W_out + w] = 0;

                    for (c = 0; c < C; c++)
                        for (p = 0; p < K; p++)
                            for (q = 0; q < K; q++)
                                Y[n * M * H_out * W_out + m * H_out * W_out + h * W_out + w] +=
                                    X[n * C * H * W + c * H * W + (h + p) * W + (w + q)] * Wk[m * C * K * K + c * K * K + p * K + q];
                }
}
