#include <stdio.h>
#include <math.h>

void subsample_ind_and_labels_cpu(int *d_ind_sub, const int *d_ind, unsigned int *d_label_sub, const unsigned int *d_label, int n_out, float inv_sub_factor) {
    for (int ind_out = 0; ind_out < n_out; ind_out++) {
        int ind_in = (int)floorf((float)(ind_out) * inv_sub_factor);
        d_ind_sub[ind_out] = d_ind[ind_in];
        d_label_sub[ind_out] = d_label[ind_in];
    }
}

int main() {
    // 示例数据
    const int n_out = 3;
    float inv_sub_factor = 0.5;
    int d_ind[] = {1, 2, 3, 4, 5, 6};
    unsigned int d_label[] = {10, 20, 30, 40, 50, 60};
    int d_ind_sub[n_out];
    unsigned int d_label_sub[n_out];

    // 调用 subsample_ind_and_labels_cpu 函数
    subsample_ind_and_labels_cpu(d_ind_sub, d_ind, d_label_sub, d_label, n_out, inv_sub_factor);

    // 打印输出结果，这里只是一个例子，实际应用中可以根据需要进行处理
    printf("Resultant subsampled indices and labels:\n");
    for (int i = 0; i < n_out; i++) {
        printf("Index: %d, Label: %u\n", d_ind_sub[i], d_label_sub[i]);
    }

    return 0;
}
