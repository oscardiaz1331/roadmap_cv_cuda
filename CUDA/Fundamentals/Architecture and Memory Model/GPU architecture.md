# GPU Architecture (SMs, Warps, Cores)

---

## 1. Introduction to GPU Architecture

Modern GPUs are massively parallel processors designed to handle thousands of concurrent threads. Unlike CPUs which are optimized for low-latency sequential execution, GPUs are optimized for high-throughput parallel workloads — making them ideal for graphics rendering and data-parallel tasks like image processing, simulations, and machine learning.

---

## 2. Key Components of the GPU Architecture

Understanding the GPU's internal structure is essential for writing optimized CUDA programs. Below, we explore the most critical architectural elements in depth: Streaming Multiprocessors (SMs), CUDA Cores, and Warps.

### 2.1 Streaming Multiprocessors (SMs)

A **Streaming Multiprocessor (SM)** is the core unit of computation in NVIDIA GPUs. Every SM executes thousands of threads simultaneously using its internal resources. Each SM acts as a mini-processor within the GPU, handling the scheduling, instruction dispatch, and execution of threads.

#### Main components inside an SM:
- **CUDA Cores (aka Scalar Processors):** These perform basic arithmetic and logic operations (e.g., `+`, `-`, `*`, `/`) on integers and floats.
- **Special Function Units (SFUs):** These handle complex operations like trigonometric functions (`sin`, `cos`, `exp`) or reciprocal square roots.
- **Tensor Cores (on Volta and newer):** Specialized units for high-throughput matrix operations, mainly used for deep learning workloads (e.g., matrix multiply-accumulate).
- **Load/Store Units:** These handle memory accesses to shared memory, global memory, and texture memory.
- **Instruction Scheduler & Dispatch Units:** Decide which warp to run and issue instructions to CUDA cores or SFUs.
- **Registers:** Fast, thread-private storage used for local variables. Each SM has a limited number of registers.
- **Shared Memory:** Low-latency memory shared between threads in a block. Often used to speed up memory access patterns.
- **L1 Cache / Shared Memory (configurable):** Acts as a cache for global memory accesses and/or shared memory for intra-block communication.

> Example: An SM in the Ampere architecture (e.g., RTX 3060 Ti) contains:
> - 128 CUDA cores
> - 4 Tensor Cores
> - Several SFUs
> - 64 KB configurable Shared Memory / L1 Cache
> - Tens of thousands of 32-bit registers

Each SM can manage multiple **warps** (groups of 32 threads) simultaneously. Warp schedulers inside the SM ensure high throughput by overlapping memory latency with instruction execution.

### 2.2 CUDA Cores

CUDA Cores are often compared to CPU cores, but they are much simpler and more lightweight. A CUDA core can execute one instruction per cycle for a single thread.

#### Characteristics:
- A CUDA core handles basic **scalar** arithmetic operations for a single thread.
- Unlike a CPU core, it has **no control logic** or instruction fetch unit — these are managed at the SM level.
- CUDA cores operate under the **SIMT** (Single Instruction, Multiple Threads) model. This means all 32 threads in a warp execute the same instruction concurrently.

> Think of CUDA cores as "execution units" that follow instructions issued by the warp scheduler.

#### Architecture-specific example (Ampere SM):
- Each SM has **128 CUDA cores**, organized into **four processing blocks**, each with:
  - 32 CUDA cores (matching the warp size)
  - Independent scheduling units (4 warp schedulers per SM)
- The SM can execute **4 warps in parallel**, dispatching one instruction per warp per cycle.

This structure allows each SM to issue up to **4 instructions per cycle** to different warps, maximizing instruction-level parallelism.

### 2.3 Warps (Execution Unit)

A **warp** is the fundamental execution unit in CUDA. It consists of **32 threads**, which are scheduled and executed in lockstep by the SM.

#### Key details:
- Each warp shares a **program counter** — meaning all threads in the warp execute the same instruction.
- If threads in a warp take **different code paths** (e.g., due to an `if-else` statement), warp **divergence** occurs, and branches are serialized, reducing performance.
- Warp-level parallelism is essential for performance. Efficient kernels avoid divergence and keep warps "in sync."

#### Warp scheduling:
- Each SM has **multiple warp schedulers**, often 4, allowing multiple warps to be active and interleaved in execution.
- The warp scheduler chooses which warp to run based on availability of data, instruction readiness, and memory dependencies.

> GPU threads are launched in blocks, but **warps are the true unit of execution** within SMs.

---

## 3. Execution Model

- The **GPU scheduler** maps thread blocks to SMs.
- Within each SM, **warps** are scheduled and executed.
- A warp is the minimum scheduling unit.
- All 32 threads in a warp execute the same instruction at the same time (unless there is divergence).

---

## 4. Summary

| Concept        | Description |
|----------------|-------------|
| SM (Streaming Multiprocessor) | Core processing unit of the GPU |
| CUDA Core      | Executes a single thread in a warp |
| Warp           | Group of 32 threads executed in lockstep |
| SIMT Model     | Single Instruction, Multiple Threads |

---

