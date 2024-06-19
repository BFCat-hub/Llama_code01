#include <stdio.h>

void kernelMaximum(float *maxhd, float *maxvd, int start, int size) {
    int tx = start;
    float max_hd = 1.175494351e-38F;
    float max_vd = 1.175494351e-38F;

    for (; tx < size; tx++) {
        if (maxhd[tx] > max_hd)
            max_hd = maxhd[tx];
        if (maxvd[tx] > max_vd)
            max_vd = maxvd[tx];
    }
}

int main() {
    // 示例数据
    const int size = 5;
    float maxhd[size] = {1.0, 2.0, 3.0, 4.0, 5.0};
    float maxvd[size] = {5.0, 4.0, 3.0, 2.0, 1.0};

    // 调用 kernelMaximum 函数
    kernelMaximum(maxhd, maxvd, 0, size);

    // 打印输出结果，这里只是一个例子，实际应用中可以根据需要进行处理
    printf("Max_hd: %f\n", maxhd[0]);
    printf("Max_vd: %f\n", maxvd[0]);

    return 0;
}
