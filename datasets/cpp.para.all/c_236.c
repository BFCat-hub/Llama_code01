#include <stdio.h>

// Function declaration
void setOffset_cpu(int *offset, int dims, int batchSize);

int main() {
    // Example data
    const int dims = 3;
    const int batchSize = 4;
    int offset[5]; // Assuming batchSize + 1 elements

    // Function call
    setOffset_cpu(offset, dims, batchSize);

    // Output result
    printf("Resultant offset array:\n");
    for (int i = 0; i <= batchSize; i++) {
        printf("%d ", offset[i]);
    }

    return 0;
}

// Function definition
void setOffset_cpu(int *offset, int dims, int batchSize) {
    offset[0] = 0;

    for (int i = 1; i <= batchSize; i++) {
        offset[i] = i * dims;
    }
}
 
