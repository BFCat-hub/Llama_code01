#include <stdio.h>
#include <stdlib.h>

void devidecountInnerCPU(long Xsize, long Ysize, long Zsize, double *p, double *pn, int *pcountinner) {
    for (int tid = 0; tid < Xsize * Ysize * Zsize; tid++) {
        if (pcountinner[tid] > 1) {
            p[tid] = pn[tid] / pcountinner[tid];
            pn[tid] = 0;
        }
    }
}

int main() {
    // 设置示例数据大小
    long Xsize = 3;
    long Ysize = 3;
    long Zsize = 3;

    // 分配内存
    double *p = (double *)malloc(Xsize * Ysize * Zsize * sizeof(double));
    double *pn = (double *)malloc(Xsize * Ysize * Zsize * sizeof(double));
    int *pcountinner = (int *)malloc(Xsize * Ysize * Zsize * sizeof(int));

    // 填充示例输入数据，这里只是一个例子，实际应用中需要根据具体情况进行初始化
    for (int i = 0; i < Xsize * Ysize * Zsize; i++) {
        p[i] = i + 1;
        pn[i] = 2 * i;
        pcountinner[i] = i % 3;  // 假设某些元素的计数大于1
    }

    // 调用函数进行处理
    devidecountInnerCPU(Xsize, Ysize, Zsize, p, pn, pcountinner);

    // 打印输出结果，这里只是一个例子，实际应用中可以根据需要进行处理
    printf("p: ");
    for (int i = 0; i < Xsize * Ysize * Zsize; i++) {
        printf("%f ", p[i]);
    }

    printf("\npn: ");
    for (int i = 0; i < Xsize * Ysize * Zsize; i++) {
        printf("%f ", pn[i]);
    }

    // 释放内存
    free(p);
    free(pn);
    free(pcountinner);

    return 0;
}
