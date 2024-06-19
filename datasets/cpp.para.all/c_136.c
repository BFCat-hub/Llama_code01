#include <stdio.h>
#include <stdlib.h>

void add_sources_d(const float *const model, float *wfp, const float *const source_amplitude,
                   const int *const sources_z, const int *const sources_x,
                   const int nz, const int nx, const int nt, const int ns, const int it) {
    int x, b;

    for (x = 0; x < nx; x++) {
        for (b = 0; b < ns; b++) {
            int i = sources_z[b * ns + x] * nx + sources_x[b * ns + x];
            int ib = b * nz * nx + i;
            wfp[ib] += source_amplitude[b * ns * nt + x * nt + it] * model[i];
        }
    }
}

int main() {
    // 示例数据
    const int nz = 3;
    const int nx = 4;
    const int nt = 5;
    const int ns = 2;
    const int it = 3;

    float *model = (float *)malloc(nz * nx * sizeof(float));
    float *wfp = (float *)malloc(ns * nz * nx * sizeof(float));
    float *source_amplitude = (float *)malloc(ns * nt * nx * sizeof(float));
    int *sources_z = (int *)malloc(ns * nx * sizeof(int));
    int *sources_x = (int *)malloc(ns * nx * sizeof(int));

    // 初始化示例数据（这里只是一个例子，实际应用中需要根据需要初始化数据）
    for (int i = 0; i < nz * nx; ++i) {
        model[i] = i + 1.0;
    }

    for (int i = 0; i < ns * nz * nx; ++i) {
        wfp[i] = 0.0;
    }

    for (int i = 0; i < ns * nt * nx; ++i) {
        source_amplitude[i] = i + 0.5;
    }

    for (int i = 0; i < ns * nx; ++i) {
        sources_z[i] = i % nz;
        sources_x[i] = i % nx;
    }

    // 调用 add_sources_d 函数
    add_sources_d(model, wfp, source_amplitude, sources_z, sources_x, nz, nx, nt, ns, it);

    // 打印输出结果，这里只是一个例子，实际应用中可以根据需要进行处理
    for (int i = 0; i < ns * nz * nx; ++i) {
        printf("Resultant wfp[%d]: %f\n", i, wfp[i]);
    }

    // 释放动态分配的内存
    free(model);
    free(wfp);
    free(source_amplitude);
    free(sources_z);
    free(sources_x);

    return 0;
}
