#include <stdio.h>

void getDRho(const int numOfNucl, const double *psi, const double **dpsi, const double *occNo, double *drho, const char debug);

int main() {
    // 在这里可以创建测试用的数据，并调用 getDRho 函数
    // 例如：
    int numOfNucl = 3;
    double psi[] = {1.0, 2.0, 3.0};
    double *dpsi[numOfNucl];
    for (int i = 0; i < numOfNucl; ++i) {
        dpsi[i] = new double[3]; // Assuming dpsi is a 2D array
        for (int j = 0; j < 3; ++j) {
            dpsi[i][j] = i + j + 1.0;
        }
    }
    double occNo[] = {0.5, 0.7, 0.9};
    double drho[3];
    char debug = 1; // Set to 1 for debug, 0 otherwise

    // 调用函数
    getDRho(numOfNucl, psi, (const double **)dpsi, occNo, drho, debug);

    // 在这里可以添加打印结果或其他操作
    // 例如：
    printf("DRHO: %f %f %f\n", drho[0], drho[1], drho[2]);

    // 释放动态分配的内存
    for (int i = 0; i < numOfNucl; ++i) {
        delete[] dpsi[i];
    }

    return 0;
}

void getDRho(const int numOfNucl, const double *psi, const double **dpsi, const double *occNo, double *drho, const char debug) {
    drho[0] = 0;
    drho[1] = 0;
    drho[2] = 0;

    for (int i = 0; i < numOfNucl; ++i) {
        drho[0] = drho[0] + 2 * occNo[i] * psi[i] * dpsi[i][0];
        drho[1] = drho[1] + 2 * occNo[i] * psi[i] * dpsi[i][1];
        drho[2] = drho[2] + 2 * occNo[i] * psi[i] * dpsi[i][2];
    }

    if (debug == 1) {
        printf("DEBUG ▁ print ▁ of ▁ DRHO:\n");
        printf("\t%f\t%f\t%f\n", drho[0], drho[1], drho[2]);
        printf("This ▁ is ▁ the ▁ last ▁ line ( DRHO ).\n\n");
    }
}
