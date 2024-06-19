#include <stdio.h>

// 函数声明
void Init(const long long size, const double *in, double *out);

int main() {
    // 示例数据
    const long long size = 5;
    double in[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    double out[size];

    // 调用函数
    Init(size, in, out);

    // 输出结果
    printf("Array after initialization:\n");
    for (long long i = 0; i < size; i++) {
        printf("%f ", out[i]);
    }

    return 0;
}

// 函数定义
void Init(const long long size, const double *in, double *out) {
    for (long long i = 0; i < size; i++) {
        out[i] = in[i];
    }
}
 
