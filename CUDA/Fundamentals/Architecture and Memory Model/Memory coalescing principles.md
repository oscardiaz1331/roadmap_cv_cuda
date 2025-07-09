# Memory Coalescing in CUDA

## What is Memory Coalescing?

Memory coalescing is a performance optimization in CUDA where memory transactions between global memory and GPU threads are grouped efficiently. When threads in a warp access global memory in a sequential and aligned pattern, the memory controller can merge these accesses into a single memory transaction. This reduces memory latency and improves bandwidth usage.

## Why Coalescing Matters

Global memory has high latency. If each thread in a warp performs its own memory transaction, this results in multiple expensive accesses. With coalescing, those accesses are combined, greatly improving performance. Efficient memory coalescing is especially important in memory-bound kernels.

## Access Pattern for Coalescing

For devices with compute capability ≥ 2.0, memory accesses by threads in a warp are coalesced when:

- Threads access **contiguous memory locations**
- Data accesses are **properly aligned** (e.g., 32-, 64-, or 128-byte segments)
- Each thread accesses **a word (usually 4 bytes)**

This results in one or two memory transactions per warp instead of 32.

## Example: Coalesced Access Pattern

Each thread accesses the next float in a contiguous array:

```cpp
__global__ void coalescedAccess(float* input, float* output) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    output[idx] = input[idx];
}
```
## Example: Uncoalesced Access Pattern

Each thread accesses a strided location in memory (common in column-wise operations):

```cpp
__global__ void uncoalescedAccess(float* input, float* output, int stride) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    output[idx] = input[idx * stride];
}
```
## Tips for Memory Coalescing

### 1. Use Structure of Arrays (SoA) Instead of Array of Structures (AoS)

**Why:**  
With AoS, threads in a warp accessing the same field of different structures will read from scattered locations in memory, which leads to uncoalesced accesses. SoA stores each field in its own array, allowing threads to access consecutive memory locations.

**Structure of Arrays (SoA) — Coalesced Access**
```cpp
struct Particles {
    float* x;
    float* y;
    float* z;
    float* velocity;
};

Particles p;
// Each thread accesses p.velocity[i] — coalesced
```
**Array of Structures (AoS) — Uncoalesced Access**

```cpp
struct Particle {
    float x, y, z;
    float velocity;
};

Particle* particles = new Particle[N];
// Each thread accesses particles[i].velocity — uncoalesced
```

### 2. Align Data Properly

Proper alignment ensures that memory accesses are aligned with the boundaries of memory transactions. This minimizes the number of memory transactions, improving performance by utilizing the global memory bandwidth more effectively.

* Using `__align__` for Custom Data Alignment

  * CUDA provides the `__align__` qualifier to align a structure's data at a specific boundary. For example, aligning data to 16 bytes can help improve memory access efficiency.

```cpp
struct __align__(16) AlignedData {
    float4 data;  // 16-byte aligned
};
```
> In this example, the float4 field will be aligned to a 16-byte boundary, which is essential for ensuring coalesced memory accesses on architectures that require such alignment.

* Using `cudaMallocPitch` for 2D Memory Allocations
  * When allocating memory for 2D arrays, it’s important to use `cudaMallocPitch`. This function ensures that rows of data are aligned to the memory pitch, improving access patterns.
```cpp
float* d_data;
size_t pitch;
cudaMallocPitch(&d_data, &pitch, width * sizeof(float), height);
```
> In this example, cudaMallocPitch allocates memory for a 2D array, ensuring that the start of each row is aligned to a multiple of 128 bytes (or another architecture-specific value). This helps with coalesced memory accesses for 2D data.

## Coalescing Requirements

For Modern CUDA Architectures (SM 2.0+ and later)

While modern CUDA architectures are more forgiving than earlier generations, performance is still best when memory accesses are:

- **Contiguous** (e.g., `input[i]`, `input[i+1]`, ...)
- **Aligned to memory segment boundaries**
- **Free of shared memory bank conflicts**

Warp coalescing is more flexible today, but ideal access patterns still matter.

---

## Stride Access Patterns

| Stride | Access Pattern               | Coalescing Efficiency     |
|--------|------------------------------|----------------------------|
| 1      | Contiguous (`A[0]`, `A[1]`)   | ✅ Excellent               |
| 2      | Every 2nd element (`A[0]`, `A[2]`) | ❌ Poor (multiple transactions) |
| N      | Strided access                | ❌ Worst                   |

---

## Matrix Access Example

Accessing a 2D array stored in row-major order:

```cpp
float A[N][M]; // linearized as A[i * M + j]
```
## Access Patterns and Coalescing

- **Reading rows** → coalesced  
- **Reading columns** → uncoalesced

### Tip

To read columns with coalesced access:

- Transpose the matrix, or  
- Use shared memory tiling to reorder accesses efficiently

---

## Summary: How to Achieve Coalescing

| Principle                      | Why It Helps                                |
|-------------------------------|----------------------------------------------|
| Access consecutive memory     | Enables warp-level merging of loads         |
| Align data properly           | Reduces number of memory transactions        |
| Avoid stride access           | Minimizes scattered memory fetches          |
| Use vector types (e.g., `float4`) | Fetch more data per instruction          |
| Use shared memory tiling      | Optimizes irregular or strided accesses     |

---

## Rule of Thumb

**"Each thread should access data that's next to the data accessed by neighboring threads."**

This ensures memory coalescing and maximizes the use of global memory bandwidth.

## Summary

Memory coalescing is key to CUDA performance. It reduces the number of memory transactions and improves bandwidth. Writing memory-access patterns that promote coalescing is one of the most impactful ways to optimize your CUDA kernels.
