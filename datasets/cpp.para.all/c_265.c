#include <stdio.h>

int seqTrans(float **h_in, float **h_out, int x_size, int y_size) ;

int main() {
    // Example data
    int x_size = 3;
    int y_size = 2;
    float **h_in;  // Initialize or allocate memory for h_in
    float **h_out; // Allocate memory for h_out

    // Function call
    seqTrans(h_in, h_out, x_size, y_size);

    // Output result
    printf("Transposed Matrix:\n");
    for (int y = 0; y < x_size; y++) {
        for (int x = 0; x < y_size; x++) {
            printf("%f ", h_out[y][x]);
        }
        printf("\n");
    }

    // Don't forget to free memory allocated for h_in and h_out

    return 0;
}

int seqTrans(float **h_in, float **h_out, int x_size, int y_size) {
    for (int y = 0; y < y_size; y++) {
        for (int x = 0; x < x_size; x++) {
            h_out[x][y] = h_in[y][x];
        }
    }
    return 1;
}
 
