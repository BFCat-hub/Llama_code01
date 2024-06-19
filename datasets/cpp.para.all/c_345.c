#include <stdio.h>

void returnResult_cpu(const float *box, const float *score, const int *label,
                      float *box_out, float *score_out, int *label_out,
                      float score_thr, const int dims) {
    for (int tid = 0; tid < dims; tid++) {
        if (score[tid] < score_thr) {
            score_out[tid] = 0;
            box_out[tid * 4 + 0] = -1;
            box_out[tid * 4 + 1] = -1;
            box_out[tid * 4 + 2] = -1;
            box_out[tid * 4 + 3] = -1;
            label_out[tid] = -1;
        } else {
            score_out[tid] = score[tid];
            box_out[tid * 4 + 0] = box[tid * 4 + 0];
            box_out[tid * 4 + 1] = box[tid * 4 + 1];
            box_out[tid * 4 + 2] = box[tid * 4 + 2];
            box_out[tid * 4 + 3] = box[tid * 4 + 3];
            label_out[tid] = label[tid];
        }
    }
}

int main() {
    // Test returnResult_cpu function with a simple example
    const int dims = 5;
    float box[] = {1.0, 2.0, 3.0, 4.0, 5.0, 2.0, 3.0, 4.0, 5.0, 6.0,
                   3.0, 4.0, 5.0, 6.0, 7.0, 4.0, 5.0, 6.0, 7.0, 8.0,
                   5.0, 6.0, 7.0, 8.0, 9.0};
    float score[] = {0.8, 0.6, 0.3, 0.9, 0.7};
    int label[] = {1, 2, 3, 4, 5};
    float box_out[dims * 4];
    float score_out[dims];
    int label_out[dims];

    float score_thr = 0.5;

    returnResult_cpu(box, score, label, box_out, score_out, label_out, score_thr, dims);

    // Display the results
    printf("Results after filtering:\n");
    for (int i = 0; i < dims; i++) {
        printf("Box: %.2f %.2f %.2f %.2f, Score: %.2f, Label: %d\n",
               box_out[i * 4], box_out[i * 4 + 1], box_out[i * 4 + 2], box_out[i * 4 + 3],
               score_out[i], label_out[i]);
    }

    return 0;
}
 
