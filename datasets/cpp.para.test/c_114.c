#include <stdio.h>

void mmul_cpu(const float *A, const float *B, float *C, int r1, int c1, int r2, int c2) {
    for (int idx = 0; idx < c2; idx++) {
        for (int idy = 0; idy < r1; idy++) {
            float temp = 0;
            for (int i = 0; i < c1; i++) {
                temp += A[idy * c1 + i] * B[i * c2 + idx];
            }
            C[idy * c2 + idx] = temp;
        }
    }
}

int main() {
    // 示例数据
    const int r1 = 2;
    const int c1 = 3;
    const int r2 = 3;
    const int c2 = 2;
    float A[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    float B[] = {7.0, 8.0, 9.0, 10.0, 11.0, 12.0};
    float C[r1 * c2];

    // 调用 mmul_cpu 函数
    mmul_cpu(A, B, C, r1, c1, r2, c2);

    // 打印输出结果，这里只是一个例子，实际应用中可以根据需要进行处理
    printf("Resultant matrix C:\n");
    for (int i = 0; i < r1; i++) {
        for (int j = 0; j < c2; j++) {
            printf("%f ", C[i * c2 + j]);
        }
        printf("\n");
    }

    return 0;
}
