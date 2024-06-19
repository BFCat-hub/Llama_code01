#include <stdio.h>

void cpu_set_sg(int *sxz, int sxbeg, int szbeg, int jsx, int jsz, int ns, int npml, int nnz) {
    for (int id = 0; id < ns; id++) {
        sxz[id] = nnz * (sxbeg + id * jsx + npml) + (szbeg + id * jsz + npml);
    }
}

int main() {
    // 示例用法
    int ns = 3;   // 你的 ns 值
    int *sxz = new int[ns];

    // 调用函数
    cpu_set_sg(sxz, 1, 2, 3, 4, ns, 5, 6);

    // 打印结果
    printf("初始化后的 sxz 数组：\n");
    for (int id = 0; id < ns; id++) {
        printf("%d ", sxz[id]);
    }

    // 释放内存
    delete[] sxz;

    return 0;
}
