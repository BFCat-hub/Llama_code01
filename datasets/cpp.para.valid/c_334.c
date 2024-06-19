#include <stdio.h>

void permuteData2_cpu(const float *input, float *output, int num, int devideNum, int featureSize, int priorNum, int batchSize);

int main() {
    // Example dimensions
    int num = 2;
    int devideNum = 3;
    int featureSize = 4;
    int priorNum = 2;
    int batchSize = 2;

    // Example input data
    float input[48] = {
        // Batch 1
        1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8,
        2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8,
        3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8,
        // Batch 2
        4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8,
        5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8,
        6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8,
    };

    // Output data
    float output[48];

    // Applying permuteData2_cpu
    permuteData2_cpu(input, output, num, devideNum, featureSize, priorNum, batchSize);

    // Print the result
    printf("Output data:\n");
    for (int s = 0; s < batchSize; ++s) {
        for (int i = 0; i < priorNum; ++i) {
            for (int j = 0; j < devideNum; ++j) {
                for (int tid = 0; tid < num; ++tid) {
                    printf("%8.4f ", output[s * num * priorNum * devideNum + tid * priorNum * devideNum + i * devideNum + j]);
                }
            }
        }
        printf("\n");
    }

    return 0;
}

void permuteData2_cpu(const float *input, float *output, int num, int devideNum, int featureSize, int priorNum, int batchSize) {
    for (int tid = 0; tid < num; tid++) {
        int numPerbatch = num * devideNum * priorNum;
        for (int s = 0; s < batchSize; s++) {
            for (int i = 0; i < priorNum; i++) {
                for (int j = 0; j < devideNum; j++) {
                    output[s * numPerbatch + tid * priorNum * devideNum + i * devideNum + j] =
                        input[s * numPerbatch + (i * devideNum * featureSize) + (j * featureSize) + tid];
                }
            }
        }
    }
}
 
