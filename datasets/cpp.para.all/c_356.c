#include <stdio.h>

void insert_sort(int a[], int n) {
    for (int i = 1; i < n; i++) {
        int j = 0;
        while ((a[j] < a[i]) && (j < i)) {
            j++;
        }
        if (i != j) {
            int temp = a[i];
            for (int k = i; k > j; k--) {
                a[k] = a[k - 1];
            }
            a[j] = temp;
        }
    }
}

int main() {
    // Test insert_sort function with a simple example
    int n = 5;
    int a[] = {5, 3, 1, 4, 2};

    // Display original array
    printf("Original array: ");
    for (int i = 0; i < n; i++) {
        printf("%d ", a[i]);
    }
    printf("\n");

    // Call the insert_sort function
    insert_sort(a, n);

    // Display sorted array
    printf("Sorted array: ");
    for (int i = 0; i < n; i++) {
        printf("%d ", a[i]);
    }
    printf("\n");

    return 0;
}
 
