#include <stdio.h>

// Function prototype
void get_conf_inds(const float *mlvl_conf, const float conf_thr, int *conf_inds, int dims);

int main() {
    // Example data
    int dims = 5;
    float mlvl_conf[] = {0.8, 0.6, 0.9, 0.4, 0.7};
    float conf_thr = 0.7;
    int conf_inds[dims];

    // Call the function
    get_conf_inds(mlvl_conf, conf_thr, conf_inds, dims);

    // Display the results
    printf("Confidence Indices:\n");
    for (int i = 0; i < dims; i++) {
        printf("%d ", conf_inds[i]);
    }
    printf("\n");

    return 0;
}

// Function definition
void get_conf_inds(const float *mlvl_conf, const float conf_thr, int *conf_inds, int dims) {
    for (int tid = 0; tid < dims; tid++) {
        if (mlvl_conf[tid] >= conf_thr) {
            conf_inds[tid] = 1;
        } else {
            conf_inds[tid] = -1;
        }
    }
}
 
