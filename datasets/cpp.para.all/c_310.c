#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Function prototype
void smooth_l1_cpu(int n, float *pred, float *truth, float *delta, float *error);

int main() {
    // Example data
    int n = 5;
    float *pred = (float *)malloc(n * sizeof(float));
    float *truth = (float *)malloc(n * sizeof(float));
    float *delta = (float *)malloc(n * sizeof(float));
    float *error = (float *)malloc(n * sizeof(float));

    // Initialize input data (for example)
    for (int i = 0; i < n; i++) {
        pred[i] = i + 1.0; // Replace with your data
        truth[i] = i + 1.5; // Replace with your data
    }

    // Call the function
    smooth_l1_cpu(n, pred, truth, delta, error);

    // Display the results
    printf("Smooth L1 Error:\n");
    for (int i = 0; i < n; i++) {
        printf("%.2f ", error[i]);
    }

    // Free allocated memory
    free(pred);
    free(truth);
    free(delta);
    free(error);

    return 0;
}

// Function definition
void smooth_l1_cpu(int n, float *pred, float *truth, float *delta, float *error) {
    int i;

    for (i = 0; i < n; ++i) {
        float diff = truth[i] - pred[i];
        float abs_val = fabs(diff);

        if (abs_val < 1) {
            error[i] = diff * diff;
            delta[i] = diff;
        } else {
            error[i] = 2 * abs_val - 1;
            delta[i] = (diff < 0) ? 1 : -1;
        }
    }
}
 
