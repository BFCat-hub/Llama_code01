#include <stdio.h>

void get_positive_data_cpu(const float *all_box, const float *all_scores, const float *all_conf, const int *conf_inds, float *positive_box, float *positive_scores, float *positive_conf, int dims, int clsNum) {
    for (int tid = 0; tid < dims; tid++) {
        if (conf_inds[tid] != (-1)) {
            positive_box[tid * 4 + 0] = all_box[tid * 4 + 0];
            positive_box[tid * 4 + 1] = all_box[tid * 4 + 1];
            positive_box[tid * 4 + 2] = all_box[tid * 4 + 2];
            positive_box[tid * 4 + 3] = all_box[tid * 4 + 3];
            for (int i = 0; i < clsNum; i++) {
                positive_scores[tid * clsNum + i] = all_scores[tid * clsNum + i];
            }
            positive_conf[tid] = all_conf[tid];
        } else {
            positive_box[tid * 4 + 0] = 0;
            positive_box[tid * 4 + 1] = 0;
            positive_box[tid * 4 + 2] = 0;
            positive_box[tid * 4 + 3] = 0;
            for (int i = 0; i < clsNum; i++) {
                positive_scores[tid * clsNum + i] = (-1);
            }
            positive_conf[tid] = (-1);
        }
    }
}

int main() {
    // Test get_positive_data_cpu function with a simple example
    int dims = 5;
    int clsNum = 3;

    float all_box[5 * 4]; // Assuming dims = 5
    float all_scores[5 * clsNum]; // Assuming dims = 5, clsNum = 3
    float all_conf[5];
    int conf_inds[5];

    float positive_box[5 * 4];
    float positive_scores[5 * clsNum];
    float positive_conf[5];

    // Initialize all_box, all_scores, all_conf, and conf_inds with appropriate values

    // Call the get_positive_data_cpu function
    get_positive_data_cpu(all_box, all_scores, all_conf, conf_inds, positive_box, positive_scores, positive_conf, dims, clsNum);

    // Display the results
    for (int i = 0; i < dims; i++) {
        printf("Example %d:\n", i);
        printf("Positive Box: [%f, %f, %f, %f]\n", positive_box[i * 4 + 0], positive_box[i * 4 + 1], positive_box[i * 4 + 2], positive_box[i * 4 + 3]);
        printf("Positive Scores: ");
        for (int j = 0; j < clsNum; j++) {
            printf("%f ", positive_scores[i * clsNum + j]);
        }
        printf("\nPositive Confidence: %f\n", positive_conf[i]);
        printf("\n");
    }

    return 0;
}
 
