#include <device_launch_parameters.h>
#include <stdio.h>


__global__ void Copy_List_gpu(const int element_numbers, const float *origin_list, float *list) {
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < element_numbers; i += stride) {
        list[i] = origin_list[i];
    }
}

int main() {
    int element_numbers = 1000;

    
    float *origin_list, *list;

    
    cudaSetDevice(0);

    
    float *d_origin_list, *d_list;
    cudaMalloc((void **)&d_origin_list, element_numbers * sizeof(float));
    cudaMalloc((void **)&d_list, element_numbers * sizeof(float));

    
    cudaMemcpy(d_origin_list, origin_list, element_numbers * sizeof(float), cudaMemcpyHostToDevice);

    
    int threadsPerBlock = 256;
    int blocksPerGrid = (element_numbers + threadsPerBlock - 1) / threadsPerBlock;

    
    Copy_List_gpu<<<blocksPerGrid, threadsPerBlock>>>(element_numbers, d_origin_list, d_list);

    
    cudaDeviceSynchronize();

    
    cudaMemcpy(list, d_list, element_numbers * sizeof(float), cudaMemcpyDeviceToHost);

    
    cudaFree(d_origin_list);
    cudaFree(d_list);

    return 0;
}