#include <stdio.h>

// 声明 create_p_vect 函数
void create_p_vect(float *node_info1, float *node_info2, float *p, int n_nodes_1, int n_nodes_2);

int main() {
    // 定义两个节点信息数组
    int n_nodes_1 = 3;
    int n_nodes_2 = 4;

    float node_info1[] = {0.2, 0.6, 0.8};
    float node_info2[] = {0.3, 0.7, 0.4, 0.9};

    // 定义结果数组 p
    float p[n_nodes_1 * n_nodes_2];

    // 调用 create_p_vect 函数
    create_p_vect(node_info1, node_info2, p, n_nodes_1, n_nodes_2);

    // 打印结果
    printf("Resulting Array p:\n");
    for (int i = 0; i < n_nodes_1; i++) {
        for (int j = 0; j < n_nodes_2; j++) {
            printf("%.2f ", p[i * n_nodes_2 + j]);
        }
        printf("\n");
    }

    return 0;
}

// 定义 create_p_vect 函数
void create_p_vect(float *node_info1, float *node_info2, float *p, int n_nodes_1, int n_nodes_2) {
    int tx, ty;
    float cutoff = 0.5;
    for (tx = 0; tx < n_nodes_1; tx++) {
        for (ty = 0; ty < n_nodes_2; ty++) {
            int ind = tx * n_nodes_2 + ty;
            if ((node_info1[tx] < cutoff) && (node_info2[ty] < cutoff))
                p[ind] = 0;
            else
                p[ind] = node_info1[tx] * node_info2[ty];
        }
    }
}
 
