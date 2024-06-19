#include <stdio.h>

// Function declaration
void gaussianPass(int patchSize, int dataSize, float *gaussFilter, float *data);

int main() {
    // Example data
    const int patchSize = 3;
    const int dataSize = 9;
    float gaussFilter[] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};
    float data[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};

    // Function call
    gaussianPass(patchSize, dataSize, gaussFilter, data);

    // Output result
    printf("Resultant data array:\n");
    for (int i = 0; i < dataSize; i++) {
        printf("%f ", data[i]);
    }

    return 0;
}

// Function definition
void gaussianPass(int patchSize, int dataSize, float *gaussFilter, float *data) {
    for (int i = 0; i < dataSize; i++) {
        data[i] = gaussFilter[i % (patchSize * patchSize)] * data[i];
    }
}
 
