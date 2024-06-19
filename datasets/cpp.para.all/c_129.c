#include <stdio.h>

void getRho(const int numOfNucl, const double *psi, const double *occNo, double *rho, const char debug) {
    *rho = 0;

    for (int i = 0; i < numOfNucl; ++i)
        *rho += occNo[i] * psi[i] * psi[i];

    if (debug == 1)
        printf("DEBUG: Print of RHO:\nRHO = %f\nThis is the last line (RHO).\n\n", *rho);
}

int main() {
    // 示例数据
    const int numOfNucl = 3;
    const char debug = 1;
    double psi[] = {0.1, 0.2, 0.3};
    double occNo[] = {1.0, 2.0, 3.0};
    double rho;

    // 调用 getRho 函数
    getRho(numOfNucl, psi, occNo, &rho, debug);

    // 打印输出结果，这里只是一个例子，实际应用中可以根据需要进行处理
    printf("Resultant RHO: %f\n", rho);

    return 0;
}
