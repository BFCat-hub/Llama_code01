#include <stdio.h>
#include <stdlib.h>

void *Match(int num_points, float *P, float *Q, int q_points, int *idx, int start, int end);

int main() {
    // Example parameters
    const int num_points = 3;
    const int q_points = 3;
    const int points_per_coordinate = 3;
    const int array_size = num_points * points_per_coordinate;
    const int start = 0;
    const int end = num_points;

    // Input arrays
    float P[array_size] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
    float Q[array_size] = {10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f};
    
    // Output array
    int idx[num_points];

    // Call Match function
    Match(num_points, P, Q, q_points, idx, start, end);

    // Print the results
    printf("Matching Indices:\n");
    for (int i = 0; i < num_points; ++i) {
        printf("P[%d] matches with Q[%d]\n", i, idx[i]);
    }

    return 0;
}

void *Match(int num_points, float *P, float *Q, int q_points, int *idx, int start, int end) {
    float dist;
    float max_dist;

    for (int i = start; i < end; i++) {
        max_dist = 1000000000.0f;

        for (int j = 0; j < num_points; j++) {
            dist = (P[0 + i * 3] - Q[0 + j * 3]) * (P[0 + i * 3] - Q[0 + j * 3]) +
                   (P[1 + i * 3] - Q[1 + j * 3]) * (P[1 + i * 3] - Q[1 + j * 3]) +
                   (P[2 + i * 3] - Q[2 + j * 3]) * (P[2 + i * 3] - Q[2 + j * 3]);

            if (dist < max_dist) {
                max_dist = dist;
                idx[i] = j;
            }
        }
    }

    return (void *)0;
}
