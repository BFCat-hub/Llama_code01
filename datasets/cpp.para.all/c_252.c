#include <stdio.h>
#include <omp.h>

// Function declaration
double histogram_serial(const int *values, int *bins, const int nbins, const int n);

int main() {
    // Example data
    const int n = 10;
    int values[] = {1, 2, 3, 4, 5, 1, 2, 3, 4, 5};
    const int nbins = 5;
    int bins[5];

    // Function call
    double time = histogram_serial(values, bins, nbins, n);

    // Output result
    printf("Histogram bins:\n");
    for (int i = 0; i < nbins; i++) {
        printf("Bin %d: %d\n", i, bins[i]);
    }

    // Output time
    printf("Time taken: %f seconds\n", time);

    return 0;
}

// Function definition
double histogram_serial(const int *values, int *bins, const int nbins, const int n) {
    double time = -omp_get_wtime();

    // Initialize bins to zero
    for (int i = 0; i < nbins; ++i) {
        bins[i] = 0;
    }

    // Count occurrences of each value and update bins
    for (int i = 0; i < n; ++i) {
        bins[values[i]]++;
    }

    time += omp_get_wtime();
    return time;
}
 
