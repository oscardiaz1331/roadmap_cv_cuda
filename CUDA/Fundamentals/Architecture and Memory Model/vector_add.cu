#include <cmath>
#include <cstdio>

__global__ void addKernel(float* A, float* B, float* C, int vector_size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < vector_size) {
        C[i] = A[i] + B[i];
    }
}

__global__ void addKernelFloat4(float4* A, float4* B, float4* C, int vector_size_float4)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < vector_size_float4) {
        C[i].x = A[i].x + B[i].x;
        C[i].y = A[i].y + B[i].y;
        C[i].z = A[i].z + B[i].z;
        C[i].w = A[i].w + B[i].w;
    }
}

int main(){
    int vector_size = static_cast<int>(pow(2.0, 20.0));
    float* A_host = new float[vector_size];
    float* B_host = new float[vector_size];
    float* C_host = new float[vector_size];

    for (int i = 0; i < vector_size; i++) {
        A_host[i] = 1.0f;
        B_host[i] = 2.0f;
    }

    size_t num_bytes = vector_size * sizeof(float);

    float* A_device;
    float* B_device;
    float* C_device;

    cudaMalloc((void**)&A_device, num_bytes);
    cudaMalloc((void**)&B_device, num_bytes);
    cudaMalloc((void**)&C_device, num_bytes);


    cudaMemcpy(A_device, A_host, num_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(B_device, B_host, num_bytes, cudaMemcpyHostToDevice);


    int blockSize = 256;
    int numBlocks = (vector_size + blockSize - 1) / blockSize;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    addKernel<<<numBlocks, blockSize>>>(A_device, B_device, C_device, vector_size);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Elapsed time: %f ms\n", milliseconds);
    cudaMemcpy(C_host, C_device, num_bytes, cudaMemcpyDeviceToHost);

    cudaFree(A_device);
    cudaFree(B_device);
    cudaFree(C_device);


    // Implementing a unified memory approach
    // Data is accessible from both host and device
    float* A_unified;
    float* B_unified;
    float* C_unified;

    cudaError_t errA = cudaMallocManaged((void**)&A_unified, num_bytes);
    cudaError_t errB = cudaMallocManaged((void**)&B_unified, num_bytes);
    cudaError_t errC = cudaMallocManaged((void**)&C_unified, num_bytes);

    if (errA != cudaSuccess || errB != cudaSuccess || errC != cudaSuccess) {
        fprintf(stderr, "Failed to allocate unified memory! Error: %s\n", cudaGetErrorString(cudaGetLastError()));
        return 1;
    }

    // Initialize data on the host (accessible via A_unified, B_unified pointers)
    for (int i = 0; i < vector_size; i++) {
        A_unified[i] = 1.0f;
        B_unified[i] = 2.0f;
    }

    cudaEventRecord(start, 0);
    addKernel<<<numBlocks, blockSize>>>(A_unified, B_unified, C_unified, vector_size);
    cudaEventRecord(stop, 0); 
    cudaEventSynchronize(stop);

    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Unified Memory - Elapsed kernel time: %f ms\n", milliseconds);

    cudaFree(A_unified);
    cudaFree(B_unified);
    cudaFree(C_unified);

    // Using float4 for vector addition
    int vector_size_float4 = vector_size / 4;
    size_t num_bytes_float4 = vector_size_float4 * sizeof(float4);

    float4* A_device_float4;
    float4* B_device_float4;
    float4* C_device_float4;

    cudaMalloc((void**)&A_device_float4, num_bytes_float4);
    cudaMalloc((void**)&B_device_float4, num_bytes_float4);
    cudaMalloc((void**)&C_device_float4, num_bytes_float4);

    cudaMemcpy(A_device_float4, A_host, num_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(B_device_float4, B_host, num_bytes, cudaMemcpyHostToDevice);

    cudaEventRecord(start, 0); 
    addKernelFloat4<<<numBlocks, blockSize>>>(A_device_float4, B_device_float4, C_device_float4, vector_size_float4);
    cudaEventRecord(stop, 0);  
    cudaEventSynchronize(stop); 

    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Float4 Explicit Memcpy - Elapsed kernel time: %f ms\n", milliseconds);

    cudaMemcpy(C_host, C_device_float4, num_bytes, cudaMemcpyDeviceToHost);

    delete[] A_host;
    delete[] B_host;
    delete[] C_host;
    cudaFree(A_device_float4);
    cudaFree(B_device_float4);
    cudaFree(C_device_float4);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}