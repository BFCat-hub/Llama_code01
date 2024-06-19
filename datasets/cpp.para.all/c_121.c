#include <stdio.h>

void diffusion(const double *x0, double *x1, int nx, int ny, double dt) {
    int i, j;
    auto width = nx + 2;
    for (j = 1; j < ny + 1; ++j) {
        for (i = 1; i < nx + 1; ++i) {
            auto pos = i + j * width;
            x1[pos] = x0[pos] + dt * (-4. * x0[pos] + x0[pos - width] + x0[pos + width] + x0[pos - 1] + x0[pos + 1]);
        }
    }
}

int main() {
    // 示例数据
    const int nx = 2;
    const int ny = 2;
    const double dt = 0.1;
    double x0[(nx + 2) * (ny + 2)] = {0.0}; // Assuming initial values are 0
    double x1[(nx + 2) * (ny + 2)] = {0.0};

    // 调用 diffusion 函数
    diffusion(x0, x1, nx, ny, dt);

    // 打印输出结果，这里只是一个例子，实际应用中可以根据需要进行处理
    printf("Resultant array x1:\n");
    for (int j = 0; j < ny + 2; j++) {
        for (int i = 0; i < nx + 2; i++) {
            auto pos = i + j * (nx + 2);
            printf("%f ", x1[pos]);
        }
        printf("\n");
    }

    return 0;
}
