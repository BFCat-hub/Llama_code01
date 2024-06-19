#include <stdio.h>
#include <math.h>

void colLog2SumExp2_cpu(const double *mat, double *buf, int m, int n) {
    for (int j = 0; j < n; j++) {
        double maximum = mat[j];

        for (int i = 1; i < m; i++) {
            if (mat[i * n + j] > maximum) {
                maximum = mat[i * n + j];
            }
        }

        double res = 0.0;

        for (int i = 0; i < m; i++) {
            res += exp(mat[i * n + j] - maximum);
        }

        buf[j] = log2(res) + maximum;
    }
}

int main() {
    // 示例数据
    const int m = 3;
    const int n = 2;
    double mat[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    double buf[n];

    // 调用 colLog2SumExp2_cpu 函数
    colLog2SumExp2_cpu(mat, buf, m, n);

    // 打印输出结果，这里只是一个例子，实际应用中可以根据需要进行处理
    for (int j = 0; j < n; j++) {
        printf("Resultant buf[%d]: %f\n", j, buf[j]);
    }

    return 0;
}
