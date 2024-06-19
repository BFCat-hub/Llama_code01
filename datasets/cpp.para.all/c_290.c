#include <stdio.h>

// Function prototype
void update_clusters_cpu(int n, int k, double *Cx, double *Cy, double *Cx_sum, double *Cy_sum, int *Csize);

int main() {
    // Example data
    int n = 5;
    int k = 3;
    double Cx[k] = {1.0, 2.0, 3.0};
    double Cy[k] = {4.0, 5.0, 6.0};
    double Cx_sum[k] = {10.0, 20.0, 30.0};
    double Cy_sum[k] = {40.0, 50.0, 60.0};
    int Csize[k] = {2, 3, 0}; // Assuming Csize is non-zero for non-empty clusters

    // Call the function
    update_clusters_cpu(n, k, Cx, Cy, Cx_sum, Cy_sum, Csize);

    // Display the results
    printf("Updated Clusters:\n");
    for (int index = 0; index < k; index++) {
        printf("Cluster %d: Cx=%.2f, Cy=%.2f\n", index, Cx[index], Cy[index]);
    }

    return 0;
}

// Function definition
void update_clusters_cpu(int n, int k, double *Cx, double *Cy, double *Cx_sum, double *Cy_sum, int *Csize) {
    for (int index = 0; index < k; index++) {
        if (Csize[index]) {
            Cx[index] = Cx_sum[index] / Csize[index];
            Cy[index] = Cy_sum[index] / Csize[index];
        }
    }
}
 
