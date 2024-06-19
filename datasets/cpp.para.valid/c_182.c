#include <stdio.h>

// 函数声明
void Mul_half_cpu(float *src, float *dst);

int main() {
    // 示例数据
    float src[] = {2.0, 4.0, 6.0};
    float dst[3];

    // 调用函数
    Mul_half_cpu(src, dst);

    // 输出结果
    printf("Array after multiplying each element by 0.5:\n");
    for (int i = 0; i < 3; i++) {
        printf("%f ", dst[i]);
    }

    return 0;
}

// 函数定义
void Mul_half_cpu(float *src, float *dst) {
    for (int index = 0; index < 3; index++) {
        dst[index] = src[index] * 0.5;
    }
}
 
