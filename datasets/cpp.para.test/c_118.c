#include <stdio.h>

void cpu_cross_correlate(float *Isg, float *Iss, float *sp, float *gp, int npml, int nnz, int nnx) {
    for (int i1 = npml; i1 < nnz - npml; i1++) {
        for (int i2 = npml; i2 < nnx - npml; i2++) {
            int id = i1 + i2 * nnz;
            float ps = sp[id];
            float pg = gp[id];
            Isg[id] += ps * pg;
            Iss[id] += ps * ps;
        }
    }
}

int main() {
    // 示例数据
    const int npml = 1;
    const int nnz = 4;
    const int nnx = 3;
    float sp[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0};
    float gp[] = {12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0};
    float Isg[nnz * nnx] = {0};
    float Iss[nnz * nnx] = {0};

    // 调用 cpu_cross_correlate 函数
    cpu_cross_correlate(Isg, Iss, sp, gp, npml, nnz, nnx);

    // 打印输出结果，这里只是一个例子，实际应用中可以根据需要进行处理
    printf("Resultant arrays Isg and Iss:\n");
    for (int i2 = 0; i2 < nnx; i2++) {
        for (int i1 = 0; i1 < nnz; i1++) {
            int id = i1 + i2 * nnz;
            printf("Isg[%d]: %f, Iss[%d]: %f\n", id, Isg[id], id, Iss[id]);
        }
    }

    return 0;
}
