#include <device_launch_parameters.h>
#include <stdio.h>


__global__ void gpu_record(float* p, float* seis_kt, int* Gxz, int ng) {
    int id = threadIdx.x + blockDim.x * blockIdx.x;
    if (id < ng) {
        seis_kt[id] = p[Gxz[id]];
    }
}

int main() {
    
    int ng = 1000; 

    
    float* h_p = (float*)malloc(ng * sizeof(float));
    float* h_seis_kt = (float*)malloc(ng * sizeof(float));
    int* h_Gxz = (int*)malloc(ng * sizeof(int));

    
    float* d_p, * d_seis_kt;
    int* d_Gxz;
    cudaMalloc((void**)&d_p, ng * sizeof(float));
    cudaMalloc((void**)&d_seis_kt, ng * sizeof(float));
    cudaMalloc((void**)&d_Gxz, ng * sizeof(int));

    
    cudaMemcpy(d_p, h_p, ng * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Gxz, h_Gxz, ng * sizeof(int), cudaMemcpyHostToDevice);

    
    dim3 blockSize(256); 
    dim3 gridSize((ng + blockSize.x - 1) / blockSize.x, 1);

    
    gpu_record<<<gridSize, blockSize>>>(d_p, d_seis_kt, d_Gxz, ng);

    
    cudaMemcpy(h_seis_kt, d_seis_kt, ng * sizeof(float), cudaMemcpyDeviceToHost);

    
    for (int i = 0; i < 10; ++i) {
        printf("seis_kt[%d]: %f\n", i, h_seis_kt[i]);
    }

    
    free(h_p);
    free(h_seis_kt);
    free(h_Gxz);
    cudaFree(d_p);
    cudaFree(d_seis_kt);
    cudaFree(d_Gxz);

    return 0;
}