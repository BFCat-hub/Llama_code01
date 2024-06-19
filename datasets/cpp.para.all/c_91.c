#include <stdio.h>
#include <stdlib.h>

double *ObjFeatures_circularity(const int compCount, const int *areaRes, const double *perimeter) {
    if (compCount > 0) {
        double *circ = (double *)malloc(compCount * sizeof(double));
        for (int i = 0; i < compCount; i++) {
            circ[i] = (4.0 * 3.14159265359 * (double)areaRes[i]) / (perimeter[i] * perimeter[i]);
        }
        return circ;
    }
    return (double *)0;
}

int main() {
    // 设置示例数据
    const int compCount = 3;
    int areaRes[] = {10, 15, 20};
    double perimeter[] = {12.56, 18.85, 25.13};

    // 调用函数计算圆形度特征
    double *circ = ObjFeatures_circularity(compCount, areaRes, perimeter);

    // 打印输出结果，这里只是一个例子，实际应用中可以根据需要进行处理
    printf("Circularity features:\n");
    for (int i = 0; i < compCount; i++) {
        printf("%f ", circ[i]);
    }

    // 释放内存
    free(circ);

    return 0;
}
