#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void Softmax_seg(float *x, const int size_category, const int size_spatial_feature_map) {
    int c = size_category;
    int size = size_spatial_feature_map;
    float temp1, temp2;

    for (int i = 0; i < size; i++) {
        temp1 = 0.;
        temp2 = 0.;

        for (int j = 0; j < c; j++) {
            temp1 = fmaxf(x[j * size + i], temp1);
        }

        for (int j = 0; j < c; j++) {
            x[j * size + i] = expf(x[j * size + i] - temp1);
            temp2 += x[j * size + i];
        }

        for (int j = 0; j < c; j++) {
            x[j * size + i] /= temp2;
        }
    }
}

int main() {
    // Test Softmax_seg function with a simple example
    int size_category = 3;
    int size_spatial_feature_map = 4;
    float input[] = {1.0, 2.0, 3.0, 2.0, 1.0, 3.0, 3.0, 2.0, 1.0, 2.0, 1.0, 3.0};
    float output[size_category * size_spatial_feature_map];

    printf("Input:\n");
    for (int i = 0; i < size_category * size_spatial_feature_map; i++) {
        printf("%.2f ", input[i]);
    }
    printf("\n");

    Softmax_seg(input, size_category, size_spatial_feature_map);

    printf("Output:\n");
    for (int i = 0; i < size_category * size_spatial_feature_map; i++) {
        printf("%.4f ", input[i]);
    }
    printf("\n");

    return 0;
}
 
