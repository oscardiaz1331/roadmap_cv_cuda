# CUDA Project 1b – Matrix Multiplication with Memory Optimization

## Objective

Implement a CUDA program that:

- Performs matrix multiplication C = A × B using different memory optimization techniques.
- Compares performance between naive implementation and optimized versions.
- Demonstrates the impact of shared memory and memory coalescing.
- Analyzes memory bandwidth utilization.

---

## Step-by-Step Instructions

### Step 1 – Define Matrix Dimensions

Choose square matrices for simplicity. Good starting sizes:

- Small: 512 × 512 (for debugging)
- Medium: 1024 × 1024 (for performance analysis)
- Large: 2048 × 2048 (for memory optimization impact)

**Memory calculation:** For 1024×1024 float matrices, each matrix uses 4MB, total ~12MB.

---

### Step 2 – Allocate Host Memory

You'll need three matrices on the CPU:

- Input matrix A (M × K)
- Input matrix B (K × N)
- Output matrix C (M × N)

**Tasks:**

- Allocate memory using `malloc` with proper alignment.
- Initialize A and B with random values or simple patterns (e.g., A[i][j] = i + j).
- Initialize C to zeros.

```c
// Example initialization
for(int i = 0; i < M; i++) {
    for(int j = 0; j < K; j++) {
        A[i * K + j] = (float)(i + j);
    }
}
```

---

### Step 3 – Allocate Device Memory

Allocate memory on the GPU for all three matrices.

**Tasks:**

- Use `cudaMalloc` for device versions of A, B, and C.
- Copy A and B from host to device using `cudaMemcpy`.
- Ensure proper error checking for all CUDA calls.

---

### Step 4 – Write CUDA Kernels

Implement multiple versions to compare performance:

#### Version 1: Naive Implementation

Each thread computes one element of C:

```c
__global__ void matrixMulNaive(float* A, float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}
```

#### Version 2: Shared Memory Tiled Implementation

Use shared memory to reduce global memory accesses:

```c

#define TILE_SIZE 32  // Size of tile (block of threads)

// CUDA kernel to perform matrix multiplication: C = A * B
__global__ void matrixMulTiledKernel(float* A, float* B, float* C, int M, int N, int K) 
{
    // Allocate shared memory tiles for A and B
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    // Compute the global row and column this thread is responsible for
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;  // Accumulator for the dot product

    // Loop over tiles of size TILE_SIZE along the K dimension
    for(int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; ++tile) 
    {
        // Load a tile of A into shared memory
        if (row < M && tile * TILE_SIZE + threadIdx.x < K)
            tileA[threadIdx.y][threadIdx.x] = A[row * K + tile * TILE_SIZE + threadIdx.x];
        else
            tileA[threadIdx.y][threadIdx.x] = 0.0f;

        // Load a tile of B into shared memory
        if (col < N && tile * TILE_SIZE + threadIdx.y < K)
            tileB[threadIdx.y][threadIdx.x] = B[(tile * TILE_SIZE + threadIdx.y) * N + col];
        else
            tileB[threadIdx.y][threadIdx.x] = 0.0f;

        // Wait for all threads in the block to finish loading the tiles
        __syncthreads();

        // Compute the partial dot product for this tile
        for(int k = 0; k < TILE_SIZE; ++k)
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];

        // Wait before loading the next tile
        __syncthreads();
    }

    // Write the result to global memory if within bounds
    if (row < M && col < N)
        C[row * N + col] = sum;
}
```

---

### Step 5 – Configure Launch Parameters

For different kernel versions:

#### Naive Version:
- `blockDim = dim3(16, 16)` (256 threads per block)
- `gridDim = dim3((N + 15) / 16, (M + 15) / 16)`

#### Tiled Version:
- `blockDim = dim3(TILE_SIZE, TILE_SIZE)`
- `gridDim = dim3((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE)`

---

### Step 6 – Implement CPU Reference

Create a CPU version for correctness verification:

```c
void matrixMulCPU(float* A, float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}
```

---

### Step 7 – Performance Measurement

Measure and compare all implementations:

```c
// CPU Timing
auto startCPU = std::chrono::high_resolution_clock::now();
matrixMultCPU(A_host.get(), B_host.get(), C_host.get(), M, N, K);
auto endCPU = std::chrono::high_resolution_clock::now();
std::chrono::duration<float> durationCPU = endCPU - startCPU;
std::cout << "CPU time: " << durationCPU.count() << " seconds\n"
// Block and grid sizes
dim3 blockDim(32, 32);
dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
             (M + blockDim.y - 1) / blockDim.y);
	
// Create CUDA events
cudaEvent_t start_naive, stop_naive;
cudaEvent_t start_tiled, stop_tiled;
CUDA_CHECK(cudaEventCreate(&start_naive));
CUDA_CHECK(cudaEventCreate(&stop_naive));
CUDA_CHECK(cudaEventCreate(&start_tiled));
CUDA_CHECK(cudaEventCreate(&stop_tiled));
	
// ---- LAUNCH NAIVE KERNEL ----
CUDA_CHECK(cudaEventRecord(start_naive));
matrixMulNaiveKernel<<<gridDim, blockDim>>>(
    A_device.get(), B_device.get(), C_device.get(), M, N, K);
CUDA_CHECK(cudaEventRecord(stop_naive));
CUDA_CHECK(cudaEventSynchronize(stop_naive));

float time_naive = 0.0f;
CUDA_CHECK(cudaEventElapsedTime(&time_naive, start_naive, stop_naive));
std::cout << "Naive kernel time: " << time_naive << " ms\n";

// ---- LAUNCH TILED KERNEL ----
CUDA_CHECK(cudaEventRecord(start_tiled));
matrixMulTiledKernel<<<gridDim, blockDim>>>(
    A_device.get(), B_device.get(), C_device.get(), M, N, K);
CUDA_CHECK(cudaEventRecord(stop_tiled));
CUDA_CHECK(cudaEventSynchronize(stop_tiled));

float time_tiled = 0.0f;
CUDA_CHECK(cudaEventElapsedTime(&time_tiled, start_tiled, stop_tiled));
std::cout << "Tiled kernel time: " << time_tiled << " ms\n";
```

---

### Step 8 – Calculate Performance Metrics

Compute meaningful performance indicators:

```c
// Calculate FLOPS (Floating Point Operations Per Second)
long long flops = 2LL * M * N * K;  // Each output element needs K multiply-adds
float gflops_naive = (flops / (timeNaive * 1e-3)) / 1e9;
float gflops_tiled = (flops / (timeTiled * 1e-3)) / 1e9;

// Calculate memory bandwidth utilization
long long bytes_accessed = (long long)(M * K + K * N + M * N) * sizeof(float);
float bandwidth_naive = (bytes_accessed / (timeNaive * 1e-3)) / 1e9;  // GB/s
float bandwidth_tiled = (bytes_accessed / (timeTiled * 1e-3)) / 1e9;   // GB/s
```

---

### Step 9 – Verify Correctness

Compare GPU results with CPU reference:

```c
bool verifyResult(float* cpu_result, float* gpu_result, int size) {
    const float epsilon = 1e-3f;
    for (int i = 0; i < size; i++) {
        if (fabs(cpu_result[i] - gpu_result[i]) > epsilon) {
            printf("Mismatch at index %d: CPU = %f, GPU = %f\n", 
                   i, cpu_result[i], gpu_result[i]);
            return false;
        }
    }
    return true;
}
```

---

### Step 10 – Memory Optimization Analysis

Analyze different memory access patterns:

**Metrics to measure:**
- Global memory accesses per thread
- Shared memory bank conflicts
- Memory coalescing efficiency
- Cache hit rates

**Use profiling tools:**
- `nvprof` for basic metrics
- Nsight Compute for detailed memory analysis

---

### Step 11 – Clean Up

Properly release all resources:

```c
CUDA_CHECK(cudaEventDestroy(start_naive));
CUDA_CHECK(cudaEventDestroy(stop_naive));
CUDA_CHECK(cudaEventDestroy(start_tiled));
CUDA_CHECK(cudaEventDestroy(stop_tiled));
```

---

## Performance Analysis Questions

Answer these questions based on your results:

1. **How much speedup does the tiled version achieve over the naive version?**
CPU time: 6624.34 ms
Naive kernel time: 3.76963 ms
Results match!
Tiled kernel time: 2.71498 ms
Results match!

2. **What is the memory bandwidth utilization for each version?**
CPU:    0.324181 GFLOPS, 0.0018995 GB/s
Naive:  569.68 GFLOPS, 3.33797 GB/s
Tiled:  790.977 GFLOPS, 4.63463 GB/s
3. **How does performance scale with matrix size?**

| Matrix Size (N, M, K) | Metric    | CPU               | Naive Kernel      | Tiled Kernel      | Units  |
| :-------------------- | :-------- | :---------------- | :---------------- | :---------------- | :----- |
| **128x128x128** | Time      | 2.0662            | 0.366592 | **0.099712**         | ms     |
|                       | GFLOPS    | 2.02996          | 11.4413 | **42.0642**           | GFLOPS |
|                       | Bandwidth | 0.0951544         | 0.536313 | **1.97176**            | GB/s   |
| **512x512x512** | Time      | 310.95            | **0.89296** | 0.938464          | ms     |
|                       | GFLOPS    | 0.863275          | **300.613** | 286.037           | GFLOPS |
|                       | Bandwidth | 0.0101165         | **3.52281** | 3.352             | GB/s   |
| **1024x1024x1024** | Time      | 6624.34           | 3.76963           | **2.71498** | ms     |
|                       | GFLOPS    | 0.324181          | 569.68            | **790.977** | GFLOPS |
|                       | Bandwidth | 0.0018995         | 3.33797           | **4.63463** | GB/s   |
| **2048x2048x2048** | Time      | 59483.4           | 162.51            | **22.7422** | ms     |
|                       | GFLOPS    | 0.288818          | 105.715           | **755.418** | GFLOPS |
|                       | Bandwidth | 0.000846146       | 0.309713          | **2.21314** | GB/s   |


---

## Extra Optimizations
0. **CPU:**
CPU time: 6728.88 ms
CPU: 0.319144 GFLOPS, 0.00186999 GB/s
0. **Naive:**
Naive time: 18.6143 ms
Results match!
Naive: 115.368 GFLOPS, 0.675982 GB/s
0. **Tiled:**
Tiled time: 18.5508 ms
Results match!
Tiled: 115.762 GFLOPS, 0.678295 GB/s
1. **Prefetching:** Use `__ldg()` for read-only data
Tiled_LDG time: 12.9628 ms
Results match!
Tiled_LDG: 165.665 GFLOPS, 0.970693 GB/s
2. **Register Blocking:** Compute multiple output elements per thread
RegisterBlocked time: 15.6539 ms
Results match!
RegisterBlocked: 137.185 GFLOPS, 0.80382 GB/s
3. **cuBLAS Comparison:** Compare with highly optimized library
Seems that our implementation is better. 
cuBLAS time: 42.0731 ms
Mismatch at index 0: CPU = 357389440.000000, GPU = 357389504.000000
Results do not match!
cuBLAS: 51.0417 GFLOPS, 0.299073 GB/s

---

