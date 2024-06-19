#include <stdio.h>

// Function prototype
void sumAndScale_cpu(float *noiseVariance, float *diffMag2, int n);

int main() {
    // Example data
    int n = 3;
    float diffMag2[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    float noiseVariance[n];

    // Call the function
    sumAndScale_cpu(noiseVariance, diffMag2, n);

    // Display the results
    printf("Noise Variance:\n");
    for (int i = 0; i < n; i++) {
        printf("%.8f ", noiseVariance[i]);
    }
    printf("\n");

    return 0;
}

// Function definition
void sumAndScale_cpu(float *noiseVariance, float *diffMag2, int n) {
    for (int i = 0; i < n; i++) {
        int batchJump = i * 347;
        float temp = 0;
        for (int sumIndex = 0; sumIndex < 347; sumIndex++) {
            temp += diffMag2[batchJump + sumIndex];
        }
        temp = 0.00161812 * temp;
        noiseVariance[i] = temp;
    }
}
 
