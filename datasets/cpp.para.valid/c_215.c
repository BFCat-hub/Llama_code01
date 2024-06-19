#include <stdio.h>

// 函数声明
void doubleArrayScalarSubstract_cpu(double *d_in, double *d_out, int length, double scalar);

int main() {
    // 示例数据
    int length = 5;
    double d_in[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    double d_out[5];
    double scalar = 2.0;

    // 调用函数
    doubleArrayScalarSubstract_cpu(d_in, d_out, length, scalar);

    // 输出结果
    printf("Resultant array after scalar subtraction:\n");
    for (int i = 0; i < length; i++) {
        printf("%f ", d_out[i]);
    }

    return 0;
}

// 函数定义
void doubleArrayScalarSubstract_cpu(double *d_in, double *d_out, int length, double scalar) {
    for (int idx = 0; idx < length; idx++) {
        d_out[idx] = d_in[idx] - scalar;
    }
}
 
