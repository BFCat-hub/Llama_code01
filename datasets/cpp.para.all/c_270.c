#include <stdio.h>

// Function prototype
void histogram(int *hist_out, unsigned char *img_in, int img_size, int nbr_bin);

int main() {
    // Example data
    int nbr_bin = 256;  // Assuming 8-bit image
    int img_size = 10;
    unsigned char img_in[] = {1, 2, 3, 4, 5, 1, 2, 3, 4, 5};
    int hist_out[nbr_bin];

    // Call the function
    histogram(hist_out, img_in, img_size, nbr_bin);

    // Display the histogram
    printf("Histogram:\n");
    for (int i = 0; i < nbr_bin; i++) {
        printf("%d ", hist_out[i]);
    }
    printf("\n");

    return 0;
}

// Function definition
void histogram(int *hist_out, unsigned char *img_in, int img_size, int nbr_bin) {
    // Initialize histogram bins
    for (int i = 0; i < nbr_bin; i++) {
        hist_out[i] = 0;
    }

    // Compute histogram
    for (int i = 0; i < img_size; i++) {
        hist_out[img_in[i]]++;
    }
}
 
