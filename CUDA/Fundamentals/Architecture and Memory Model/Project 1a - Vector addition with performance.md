# CUDA Project 1a – Vector Addition

## Objective

Implement a CUDA program that:

- Adds two vectors element-wise.
- Measures the performance (execution time) of the kernel.
- Optionally profiles performance using Nsight tools.

---

## Step-by-Step Instructions

### Step 1 – Define Vector Size

Choose how large the input vectors should be.  
A good starting point is 2^20 elements (i.e., 1,048,576 floats).

---

### Step 2 – Allocate Host Memory

You’ll need three vectors on the CPU:

- Input vector A
- Input vector B
- Output vector C

**Tasks:**

- Allocate memory for the vectors on the host using `malloc` or `new`.
- Initialize A and B with values (e.g., A[i] = 1.0f, B[i] = 2.0f).

---

### Step 3 – Allocate Device Memory

Allocate memory on the GPU for the same three vectors.

**You’ll need to:**

- Use `cudaMalloc` for A, B, and C on device.
- Copy A and B from host to device using `cudaMemcpy`.

### Step 4 – Write a CUDA Kernel

The kernel will do the actual addition on the GPU.

---

### Step 5 – Decide Launch Configuration

You need to define:

- `blockDim` = number of threads per block (e.g., 256)
- `gridDim` = number of blocks = ceil(N / blockDim)

---

### Step 6 – Launch the Kernel

Call your kernel with `<<<gridDim, blockDim>>>` syntax.

---

### Step 7 – Copy Result Back

Use `cudaMemcpy` to copy the result vector C from device to host.

---

### Step 8 – Measure Performance

Use CUDA Events:

- Create two events: `start`, `stop`
- Record `start` before kernel launch
- Record `stop` after kernel execution
- Use `cudaEventElapsedTime()` to get time in milliseconds

**Optional:** Compare time for different input sizes or block sizes.

---

### Step 9 – Clean Up

Don’t forget to:

- Free device memory using `cudaFree`
- Free host memory

**Also:**

- Destroy any CUDA events you created

---

## Extra (Optional)

- Implement a CPU version and compare timing.
> The CPU version takes 0.77 ms in average of several times.
> The GPU version takes 0.17 ms in average of several times.
- Try unified memory (`cudaMallocManaged`).
> The unified version takes 7 ms in average of several times.
- Try float4 instead of float for better throughput.
> The float4 version takes 0.15 ms in average of several times.