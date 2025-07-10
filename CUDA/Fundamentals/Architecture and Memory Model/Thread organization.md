# Thread Organization in CUDA

In CUDA, the execution model is hierarchical and highly parallel. Threads are organized into structures that reflect both the logical and hardware execution patterns. Understanding this organization is fundamental for writing efficient and scalable CUDA kernels.

## Key Abstractions

### 1. Thread

* The smallest unit of execution.
* Each thread executes a single instance of the kernel.
* Has its own registers and local memory.

### 2. Block

* A group of threads that execute concurrently on a single Streaming Multiprocessor (SM).
* Threads within a block can cooperate using:

  * Shared memory
  * Barrier synchronization (`__syncthreads()`)
* Identified by: `threadIdx.{x, y, z}`

### 3. Grid

* A collection of blocks that execute the kernel across the GPU.
* Blocks in a grid are independent; they do not share memory or synchronize with each other.
* Identified by: `blockIdx.{x, y, z}`

## Dimensions

CUDA supports up to 3 dimensions for both threads and blocks:

* `threadIdx.x`, `threadIdx.y`, `threadIdx.z`
* `blockIdx.x`, `blockIdx.y`, `blockIdx.z`

This allows natural mapping of data structures such as 1D arrays, 2D matrices, and 3D volumes.

## Execution Configuration

When launching a kernel, you specify the organization like this:

```cpp
myKernel<<<dim3(gridDimX, gridDimY), dim3(blockDimX, blockDimY)>>>(...);
```

Example for a 1D array:

```cpp
int N = 1024;
int threadsPerBlock = 256;
int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

myKernel<<<blocksPerGrid, threadsPerBlock>>>(...);
```

## Index Calculation

To find the global index of a thread (for example, to process an array):

```cpp
int idx = blockIdx.x * blockDim.x + threadIdx.x;
```

This index can then be used to access memory or perform computation.

## Summary Table

| Concept | Purpose                    | ID variable         |
| ------- | -------------------------- | ------------------- |
| Thread  | Smallest execution unit    | `threadIdx`         |
| Block   | Group of threads on one SM | `blockIdx`          |
| Grid    | Collection of blocks       | N/A (kernel launch) |

## Tips

* Choose block sizes that are multiples of 32 (warp size) for efficiency.
* Use 1D, 2D, or 3D structures depending on your data.
* Use shared memory within a block to reduce global memory access.
* Keep in mind hardware limits (e.g., max threads per block).

Thread organization is key to unlocking the performance of CUDA programs. Properly structuring grids and blocks helps in efficiently utilizing GPU resources and achieving memory coalescing.
