#include <stdio.h>

// 函数声明
void doubleArrayElementwiseSquare_cpu(double *d_in, double *d_out, int length);

int main() {
    // 示例数据
    int length = 5;
    double d_in[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    double d_out[5];

    // 调用函数
    doubleArrayElementwiseSquare_cpu(d_in, d_out, length);

    // 输出结果
    printf("Resultant array after elementwise square:\n");
    for (int i = 0; i < length; i++) {
        printf("%f ", d_out[i]);
    }

    return 0;
}

// 函数定义
void doubleArrayElementwiseSquare_cpu(double *d_in, double *d_out, int length) {
    for (int idx = 0; idx < length; idx++) {
        d_out[idx] = d_in[idx] * d_in[idx];
    }
}
 
