#include <stdio.h>

// 函数声明
void kernelUpdateHead(int *head, int *d_idxs_out, int n);

int main() {
    // 示例数据
    int n = 5;
    int head[10] = {0}; // Assuming head array has a size of 10
    int d_idxs_out[] = {1, 3, 5, 7, 9};

    // 调用函数
    kernelUpdateHead(head, d_idxs_out, n);

    // 输出结果
    printf("Head array after updating:\n");
    for (int i = 0; i < 10; i++) {
        printf("%d ", head[i]);
    }

    return 0;
}

// 函数定义
void kernelUpdateHead(int *head, int *d_idxs_out, int n) {
    for (int i = 0; i < n; i++) {
        head[d_idxs_out[i]] = 1;
    }
}
 
