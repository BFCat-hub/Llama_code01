#include <stdio.h>
#include <math.h>

// Function prototype
void logistic_x_ent_cpu(int n, float *pred, float *truth, float *delta, float *error);

int main() {
    // Example data
    int n = 3;
    float pred[] = {0.2, 0.8, 0.6};
    float truth[] = {0.0, 1.0, 1.0};
    float delta[n];
    float error[n];

    // Call the function
    logistic_x_ent_cpu(n, pred, truth, delta, error);

    // Display the results
    printf("Error:\n");
    for (int i = 0; i < n; i++) {
        printf("%.4f ", error[i]);
    }
    printf("\n");

    printf("Delta:\n");
    for (int i = 0; i < n; i++) {
        printf("%.4f ", delta[i]);
    }
    printf("\n");

    return 0;
}

// Function definition
void logistic_x_ent_cpu(int n, float *pred, float *truth, float *delta, float *error) {
    for (int i = 0; i < n; ++i) {
        float t = truth[i];
        float p = pred[i];
        error[i] = -t * log(p) - (1 - t) * log(1 - p);
        delta[i] = t - p;
    }
}
 
