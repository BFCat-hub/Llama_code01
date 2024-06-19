#include <stdio.h>

// 函数声明
void resetIndices_cpu(long *vec_out, const long N);

int main() {
    // 示例数据
    const long N = 5;
    long vec_out[N];

    // 调用函数
    resetIndices_cpu(vec_out, N);

    // 输出结果
    printf("Array after resetting indices:\n");
    for (long i = 0; i < N; i++) {
        printf("%ld ", vec_out[i]);
    }

    return 0;
}

// 函数定义
void resetIndices_cpu(long *vec_out, const long N) {
    for (long idx = 0; idx < N; idx++) {
        vec_out[idx] = idx;
    }
}
 
