#include <stdio.h>

void gather_points_kernel(int b, int c, int n, int m, const float *points, const int *idx, float *out) {
    for (int i = 0; i < b; i++) {
        for (int l = 0; l < c; l++) {
            for (int j = 0; j < m; j++) {
                int a = idx[i * m + j];
                out[(i * c + l) * m + j] = points[(i * c + l) * n + a];
            }
        }
    }
}

int main() {
    // 示例数据
    const int b = 2;
    const int c = 3;
    const int n = 4;
    const int m = 5;
    float points[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0};
    int idx[] = {1, 3, 0, 2, 1, 3, 0, 2, 1, 3};
    float out[b * c * m];

    // 调用 gather_points_kernel 函数
    gather_points_kernel(b, c, n, m, points, idx, out);

    // 打印输出结果，这里只是一个例子，实际应用中可以根据需要进行处理
    printf("Resultant array out:\n");
    for (int i = 0; i < b; i++) {
        for (int l = 0; l < c; l++) {
            for (int j = 0; j < m; j++) {
                printf("%f ", out[(i * c + l) * m + j]);
            }
            printf("\n");
        }
    }

    return 0;
}
