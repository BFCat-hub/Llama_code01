#include <stdio.h>
#include <stdlib.h>

void *RyT(float *R, float *T, float *P, float *Q, int start, int end) {
    for (int i = start; i < end; i++) {
        Q[0 + i * 3] = R[0 + 0 * 3] * P[0 + i * 3] + R[0 + 1 * 3] * P[1 + i * 3] +
                       R[0 + 2 * 3] * P[2 + i * 3] + T[0];
        Q[1 + i * 3] = R[1 + 0 * 3] * P[0 + i * 3] + R[1 + 1 * 3] * P[1 + i * 3] +
                       R[1 + 2 * 3] * P[2 + i * 3] + T[1];
        Q[2 + i * 3] = R[2 + 0 * 3] * P[0 + i * 3] + R[2 + 1 * 3] * P[1 + i * 3] +
                       R[2 + 2 * 3] * P[2 + i * 3] + T[2];
    }
    return (void *)0;
}

int main() {
    // Test RyT function with a simple example
    int num_points = 3;
    float R[9] = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    float T[3] = {1.0, 2.0, 3.0};
    float P[9] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    float Q[9];

    RyT(R, T, P, Q, 0, num_points);

    // Display the results
    printf("Results after RyT function:\n");
    for (int i = 0; i < num_points; i++) {
        printf("Q[%d]: %.2f %.2f %.2f\n", i,
               Q[0 + i * 3], Q[1 + i * 3], Q[2 + i * 3]);
    }

    return 0;
}
 
