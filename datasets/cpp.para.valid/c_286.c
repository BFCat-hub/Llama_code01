#include <stdio.h>

// Function prototype
void smallCorrelation_cpu(float *L, float *innerSums, int innerSumsLength);

int main() {
    // Example data
    int innerSumsLength = 5;
    float innerSums[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    float L[innerSumsLength];

    // Call the function
    smallCorrelation_cpu(L, innerSums, innerSumsLength);

    // Display the results
    printf("Correlation Array (L):\n");
    for (int u = 0; u < innerSumsLength; u++) {
        printf("%.2f ", L[u]);
    }
    printf("\n");

    return 0;
}

// Function definition
void smallCorrelation_cpu(float *L, float *innerSums, int innerSumsLength) {
    for (int u = 0; u < innerSumsLength; u++) {
        int realIdx = 2 * u;
        int imagIdx = realIdx + 1;
        L[u] = (innerSums[realIdx] * innerSums[realIdx]) + (innerSums[imagIdx] * innerSums[imagIdx]);
    }
}
 
