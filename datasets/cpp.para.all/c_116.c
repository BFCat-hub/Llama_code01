#include <stdio.h>

void Forwardsub_cpu(double *RES, double *LS, double *LW, double *LPR, int NI, int NJ, int Start, int J, int n) {
    for (int i = 0; i < n; i++) {
        int IJ = ((Start + i) * NI) + (J - (Start + i));
        RES[IJ] = (RES[IJ] - LS[IJ] * RES[IJ - 1] - LW[IJ] * RES[IJ - NJ]) * LPR[IJ];
    }
}

int main() {
    // 示例数据
    const int NI = 4;
    const int NJ = 4;
    const int Start = 1;
    const int J = 2;
    const int n = 2;
    double RES[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0};
    double LS[] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6};
    double LW[] = {2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5};
    double LPR[] = {0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16};

    // 调用 Forwardsub_cpu 函数
    Forwardsub_cpu(RES, LS, LW, LPR, NI, NJ, Start, J, n);

    // 打印输出结果，这里只是一个例子，实际应用中可以根据需要进行处理
    printf("Resultant array RES:\n");
    for (int i = 0; i < NI * NJ; i++) {
        printf("%f ", RES[i]);
    }
    printf("\n");

    return 0;
}
