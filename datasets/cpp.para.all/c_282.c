#include <stdio.h>

// Function prototype
void analysis(int D[], int L[], int R[], int N);

int main() {
    // Example data
    int N = 5;
    int D[] = {0, 1, 2, 3, 4};
    int L[] = {0, 1, 2, 3, 4};
    int R[] = {0, 1, 2, 3, 4};

    // Call the function
    analysis(D, L, R, N);

    // Display the results
    printf("Results:\n");
    printf("D: ");
    for (int i = 0; i < N; i++) {
        printf("%d ", D[i]);
    }
    printf("\n");
    printf("L: ");
    for (int i = 0; i < N; i++) {
        printf("%d ", L[i]);
    }
    printf("\n");
    printf("R: ");
    for (int i = 0; i < N; i++) {
        printf("%d ", R[i]);
    }
    printf("\n");

    return 0;
}

// Function definition
void analysis(int D[], int L[], int R[], int N) {
    int id;
    for (id = 0; id < N; id++) {
        int label = L[id];
        int ref;
        if (label == id) {
            do {
                label = R[ref = label];
            } while (ref ^ label);
            R[id] = label;
        }
    }
}
 
