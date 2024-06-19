#include <stdio.h>

inline void MulMatrixOnCPU(float *A, float *B, float *C, int nx, int ny) {
    int i, j, k;
    float sum = 0.0;
    for (i = 0; i < nx; i++) {
        for (j = 0; j < ny; j++) {
            sum = 0.0;
            for (k = 0; k < nx; k++) {
                sum = sum + A[i * nx + k] * B[k * nx + j];
            }
            C[i * nx + j] = sum;
        }
    }
}

int main() {
    // 示例数据
    const int nx = 2;
    const int ny = 2;
    float A[] = {1.0, 2.0, 3.0, 4.0};
    float B[] = {5.0, 6.0, 7.0, 8.0};
    float C[nx * nx];

    // 调用 MulMatrixOnCPU 函数
    MulMatrixOnCPU(A, B, C, nx, ny);

    // 打印输出结果，这里只是一个例子，实际应用中可以根据需要进行处理
    printf("Resultant matrix C:\n");
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < nx; j++) {
            printf("%f ", C[i * nx + j]);
        }
        printf("\n");
    }

    return 0;
}
