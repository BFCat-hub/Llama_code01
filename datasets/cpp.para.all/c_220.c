#include <stdio.h>

// 函数声明
void cpu_record(float *p, float *seis_kt, int *Gxz, int ng);

int main() {
    // 示例数据
    const int ng = 5;
    float p[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    int Gxz[] = {0, 2, 4, 1, 3};
    float seis_kt[5];

    // 调用函数
    cpu_record(p, seis_kt, Gxz, ng);

    // 输出结果
    printf("Resultant array after cpu_record:\n");
    for (int i = 0; i < ng; i++) {
        printf("%f ", seis_kt[i]);
    }

    return 0;
}

// 函数定义
void cpu_record(float *p, float *seis_kt, int *Gxz, int ng) {
    for (int id = 0; id < ng; id++) {
        seis_kt[id] = p[Gxz[id]];
    }
}
 
