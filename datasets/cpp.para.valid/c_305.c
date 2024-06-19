#include <stdio.h>

// Function prototype
void manage_adj_matrix(float **graph, int n);

int main() {
    // Example data
    int n = 3;
    float **graph = (float **)malloc(n * sizeof(float *));
    for (int i = 0; i < n; ++i) {
        graph[i] = (float *)malloc(n * sizeof(float));
        for (int j = 0; j < n; ++j) {
            graph[i][j] = i + j + 1.0; // Sample values, replace with your data
        }
    }

    // Call the function
    manage_adj_matrix(graph, n);

    // Display the results
    printf("Modified Adjacency Matrix:\n");
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            printf("%.2f ", graph[i][j]);
        }
        printf("\n");
    }

    // Free allocated memory
    for (int i = 0; i < n; ++i) {
        free(graph[i]);
    }
    free(graph);

    return 0;
}

// Function definition
void manage_adj_matrix(float **graph, int n) {
    for (int j = 0; j < n; ++j) {
        float sum = 0.0;
        for (int i = 0; i < n; ++i) {
            sum += graph[i][j];
        }
        for (int i = 0; i < n; ++i) {
            if (sum != 0.0) {
                graph[i][j] /= sum;
            } else {
                graph[i][j] = 1.0 / (float)n;
            }
        }
    }
}
 
