#include <stdio.h>

// 定义 VecSize
#define VecSize 5

// 函数声明
void vecAddCPU(double *pdbA, double *pdbB, double *pdbC);

int main() {
    // 示例数据
    double pdbA[VecSize] = {1.0, 2.0, 3.0, 4.0, 5.0};
    double pdbB[VecSize] = {5.0, 4.0, 3.0, 2.0, 1.0};
    double pdbC[VecSize];

    // 调用函数
    vecAddCPU(pdbA, pdbB, pdbC);

    // 输出结果
    printf("Resultant vector after addition:\n");
    for (int i = 0; i < VecSize; i++) {
        printf("%f ", pdbC[i]);
    }

    return 0;
}

// 函数定义
void vecAddCPU(double *pdbA, double *pdbB, double *pdbC) {
    for (int i = 0; i < VecSize; ++i) {
        pdbC[i] = pdbA[i] + pdbB[i];
    }
}
 
