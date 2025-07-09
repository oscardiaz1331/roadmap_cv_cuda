# CUDA Memory Hierarchy

Understanding CUDA's memory hierarchy is critical for writing high-performance GPU code. CUDA exposes several types of memory, each with different **scope**, **latency**, **bandwidth**, and **access patterns**.

## Levels of Memory

| Memory Type      | Location     | Scope               | Access Speed | Size       | Lifetime        |
|------------------|--------------|---------------------|---------------|------------|------------------|
| **Registers**     | On-chip      | Per-thread           | Very fast     | ~64KB/SM   | Thread lifetime  |
| **Shared Memory** | On-chip      | Per-block            | Fast          | 48KB–100KB | Block lifetime   |
| **Local Memory**  | Off-chip     | Per-thread           | Slow (global) | Unlimited  | Thread lifetime  |
| **Global Memory** | Off-chip     | All threads & blocks | Slow          | GBs        | Application-wide |
| **Constant Memory**| Off-chip (cached) | All threads   | Fast (read-only, cached) | 64KB | Application-wide |
| **Texture/Surface Memory** | Off-chip (cached) | All threads | Medium/Fast (read-only) | GBs | Application-wide |

---

## Memory Type Details

### Registers

- **Fastest** memory (1 clock cycle).
- Private to each thread.
- Used for frequently accessed variables.
- Allocated **automatically** by the compiler.
- Limited: **register spilling** moves data to slow local memory.
- Use when:
  - Storing temporary variables inside a thread.
  - You have small, frequently accessed data (e.g., loop counters, indices, constants).
- Avoid:
  - Large arrays or structs per thread (may cause register spilling).
- Example:
```cpp
__global__ void addKernel(float *a, float *b, float *c) {
    int i = threadIdx.x;
    float tempA = a[i];  // stored in register
    float tempB = b[i];  // stored in register
    c[i] = tempA + tempB;
}
```

### Shared Memory

- Located **on-chip**, shared among threads in the **same block**.
- Very fast if used correctly (typically ~100x faster than global).
- Explicitly allocated by the programmer.
- Ideal for **inter-thread communication**, **tiling**, and **caching**.
- Access speed can suffer from **bank conflicts**.
- Use when:
  - Threads in a block need to **collaborate** or **share data**.
  - You’re performing **tiling** in matrix operations or stencil computations.
  - You want to **cache global memory** to reduce latency.
- Avoid:
  - Storing too much data (limited size).
  - Uncoordinated access (may cause **bank conflicts**).
- Example:
```cpp
__shared__ float tileA[BLOCK_SIZE][BLOCK_SIZE];
__shared__ float tileB[BLOCK_SIZE][BLOCK_SIZE];

tileA[threadIdx.y][threadIdx.x] = A[row * width + k];
tileB[threadIdx.y][threadIdx.x] = B[k * width + col];
__syncthreads();
```
Shared memory improves performance by loading chunks of global memory only once per block and reusing them.

### Local Memory

- **Confusing name**: not local in terms of physical location.
- Actually stored in **global memory**.
- Used automatically by the compiler when:
  - There aren’t enough registers.
  - You use large arrays inside a thread.
- Very **slow** (high latency, like global).
- Use when:
  - You need large per-thread arrays or structs.
  - You run out of registers (implicitly used by compiler).
- Avoid:
  - Unless unavoidable — it is **slow**, stored in global memory.
- Example:
```cpp
__device__ void someFunc() {
    int localArray[1024];  // too large for registers → spilled to local memory
}
```
 > Compiler decides this based on register pressure. You can reduce array size or refactor to avoid spilling.

### Global Memory

- Located in **device DRAM**.
- Accessible by **all threads**.
- High latency (~400–800 cycles).
- Programmer-managed.
- Key to optimization: **coalesced memory access** (threads access adjacent memory locations).

- Use when:
  - You need large, persistent data accessible across all blocks.
  - You need to read/write data from the host.
- Avoid:
  - Frequent uncoalesced access patterns.
  - Using for temporary or intermediate values inside a block.
- Example:
```cpp
__global__ void scale(float *array, int n, float factor) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        array[i] *= factor;  // direct global memory access
}
```
 > Optimize with memory coalescing: ensure adjacent threads access adjacent addresses.

### Constant Memory

- Read-only memory stored off-chip but cached.
- Accessible by all threads.
- Best for data that:
  - Doesn't change during kernel execution.
  - Is accessed uniformly across threads.
- Max size: **64 KB**.

- Use when:
  - All threads access the same read-only values (e.g., kernel weights, constants).
  - You want to reduce bandwidth usage for uniform reads.
- Avoid:
  - If data changes frequently.
  - If threads access different elements (caching becomes ineffective).
- Example:
```cpp
__constant__ float kernel[9];

__global__ void applyKernel(float *input, float *output) {
    // kernel[] accessed uniformly → fast
}
```
 > Constant memory is cached — best when access is uniform across threads.

### Texture & Surface Memory

- Special read paths optimized for **2D/3D spatial locality**.
- Useful for image data, filtering, and interpolation.
- Often used in **computer vision** and **graphics**.
- Read-only (texture) or read-write (surface).

- Use when:
  - You are working with **images, textures**, or **spatial data**.
  - You want hardware-accelerated **interpolation** or **filtering**.
  - You need **2D locality**-optimized caching.
- Avoid:
  - Non-spatial access or write-heavy operations (texture = read-only).
- Example:
```cpp
texture<float, 2, cudaReadModeElementType> tex;

__global__ void processImage(float *output, int width, int height) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    output[y * width + x] = tex2D(tex, x, y);
}
```
 > Great for image processing and filtering. Uses spatial caching.

---

## Performance Comparison

| Memory Type      | Latency | Bandwidth | Notes                        |
|------------------|---------|-----------|------------------------------|
| Registers         | 1 cycle | Very High | Fastest, limited quantity    |
| Shared Memory     | ~1–2 cycles | High | Bank conflicts can reduce speed |
| Local Memory      | ~400–800 cycles | Low  | Stored in global memory      |
| Global Memory     | ~400–800 cycles | Low  | Optimize with coalescing     |
| Constant Memory   | ~1 cycle (if cached) | Medium | Good for uniform access     |
| Texture Memory    | ~20–100 cycles (cached) | Medium | Good spatial locality       |

---

## Optimization Tips

- Prefer **registers** for private variables.
- Use **shared memory** for intra-block communication and caching.
- **Avoid local memory** by managing register use and avoiding large arrays per thread.
- Access **global memory** efficiently:
  - Use **coalesced** access patterns.
  - Minimize frequency of access.
- Use **constant memory** for read-only, commonly used values.
- Use **texture memory** for image-related access patterns with spatial locality.

---

## Scope and Lifetime Summary

- Thread     -> Registers, Local Memory
- Block      -> Shared Memory
- Grid       -> Global, Constant, Texture
- Persistent -> Global, Constant, Texture