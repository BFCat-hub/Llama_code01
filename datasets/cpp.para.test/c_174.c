#include <stdio.h>

void get_boxes_for_nms_cpu(const float *boxes_before_nms, const float *offset, float *boxes_for_nms, int dims);

int main() {
    // Example parameters
    const int dims = 5;  // Set the appropriate value for dims
    float boxes_before_nms[dims * 4];  // Replace with your actual data
    float offset[dims];  // Replace with your actual data
    float boxes_for_nms[dims * 4];

    // Initialize your input data here (replace with your actual data)
    for (int i = 0; i < dims * 4; ++i) {
        boxes_before_nms[i] = i + 1;
    }
    for (int i = 0; i < dims; ++i) {
        offset[i] = 0.5;
    }

    // Call get_boxes_for_nms_cpu function
    get_boxes_for_nms_cpu(boxes_before_nms, offset, boxes_for_nms, dims);

    // Print the results or add further processing as needed
    for (int i = 0; i < dims * 4; ++i) {
        printf("%f ", boxes_for_nms[i]);
    }

    return 0;
}

void get_boxes_for_nms_cpu(const float *boxes_before_nms, const float *offset, float *boxes_for_nms, int dims) {
    for (int tid = 0; tid < dims; tid++) {
        if (boxes_before_nms[tid * 4] == -1 && boxes_before_nms[tid * 4 + 1] == -1 &&
            boxes_before_nms[tid * 4 + 2] == -1 && boxes_before_nms[tid * 4 + 3] == -1) {
            boxes_for_nms[tid * 4] = -1;
            boxes_for_nms[tid * 4 + 1] = -1;
            boxes_for_nms[tid * 4 + 2] = -1;
            boxes_for_nms[tid * 4 + 3] = -1;
        } else {
            boxes_for_nms[tid * 4] = boxes_before_nms[tid * 4] + offset[tid];
            boxes_for_nms[tid * 4 + 1] = boxes_before_nms[tid * 4 + 1] + offset[tid];
            boxes_for_nms[tid * 4 + 2] = boxes_before_nms[tid * 4 + 2] + offset[tid];
            boxes_for_nms[tid * 4 + 3] = boxes_before_nms[tid * 4 + 3] + offset[tid];
        }
    }
}
