#include <stdio.h>

// Function declaration
void setIndexYolov3_cpu(int *input, int dims, int batchSize);

int main() {
    // Example data
    const int dims = 3;
    const int batchSize = 2;
    int input[6]; // Assuming dims * batchSize elements

    // Function call
    setIndexYolov3_cpu(input, dims, batchSize);

    // Output result
    printf("Resultant input array:\n");
    for (int i = 0; i < batchSize; i++) {
        for (int j = 0; j < dims; j++) {
            printf("%d ", input[i * dims + j]);
        }
        printf("\n");
    }

    return 0;
}

// Function definition
void setIndexYolov3_cpu(int *input, int dims, int batchSize) {
    for (int tid = 0; tid < dims; tid++) {
        for (int i = 0; i < batchSize; i++) {
            input[i * dims + tid] = tid;
        }
    }
}
 
