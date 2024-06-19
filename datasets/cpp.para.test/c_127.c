#include <stdio.h>

void Backwardsub(double *U, double *RES, double *UN, double *UE, double *LPR, int NI, int NJ, int End, int J, int n) {
    for (int i = 0; i < n; i++) {
        int IJ = ((End - i) * NI) + (J - (End - i));
        RES[IJ] = RES[IJ] - UN[IJ] * RES[IJ + 1] - UE[IJ] * RES[IJ + NJ];
        U[IJ] = U[IJ] + RES[IJ];
    }
}

int main() {
    // 示例数据
    const int NI = 3;
    const int NJ = 3;
    const int End = 1;
    const int J = 1;
    const int n = 1;
    double U[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    double RES[] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double UN[] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    double UE[] = {2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0};
    double LPR[] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    // 调用 Backwardsub 函数
    Backwardsub(U, RES, UN, UE, LPR, NI, NJ, End, J, n);

    // 打印输出结果，这里只是一个例子，实际应用中可以根据需要进行处理
    printf("Resultant array U:\n");
    for (int i = 0; i < NI; i++) {
        for (int j = 0; j < NJ; j++) {
            printf("%f ", U[i * NJ + j]);
        }
        printf("\n");
    }

    return 0;
}
