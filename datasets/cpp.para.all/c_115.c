#include <stdio.h>

void Dot(float *C, float *A, float *B, const int r, const int c, const int n) {
    float temp;
    for (int i = 0; i < r; i++) {
        for (int j = 0; j < c; j++) {
            temp = 0.0;
            for (int k = 0; k < n; k++) {
                temp += A[i * n + k] * B[k * c + j];
            }
            C[i * c + j] = temp;
        }
    }
}

int main() {
    // 示例数据
    const int r = 2;
    const int c = 2;
    const int n = 3;
    float A[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    float B[] = {7.0, 8.0, 9.0, 10.0, 11.0, 12.0};
    float C[r * c];

    // 调用 Dot 函数
    Dot(C, A, B, r, c, n);

    // 打印输出结果，这里只是一个例子，实际应用中可以根据需要进行处理
    printf("Resultant matrix C:\n");
    for (int i = 0; i < r; i++) {
        for (int j = 0; j < c; j++) {
            printf("%f ", C[i * c + j]);
        }
        printf("\n");
    }

    return 0;
}
