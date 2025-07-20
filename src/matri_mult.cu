#include <cstdlib>
#include <stdlib.h>
#include <iostream>
#include <memory>
#include <cuda_runtime.h>
#include <stdexcept>
#include <chrono>
#include <cublas_v2.h>


#define CUDA_CHECK(call)                                              \
    do {                                                              \
        cudaError_t err = call;                                       \
        if (err != cudaSuccess) {                                     \
            throw std::runtime_error(cudaGetErrorString(err));        \
        }                                                             \
    } while (0)

//RAII memory management in device class
template <typename T>
class CudaBuffer 
{
public:
	explicit CudaBuffer(size_t size) : size_(size), data_(nullptr) 
	{
		CUDA_CHECK(cudaMalloc(&data_, size_));
	}

	~CudaBuffer() {
		if (data_) cudaFree(data_);
	}

	T* get() const { return data_; }
	size_t size() const { return size_; }

private:
    T* data_{nullptr};
	size_t size_{0};
};

__global__ void matrixMulNaiveKernel(float* A, float* B, float* C, int M, int N, int K) 
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < M && col < N) 
	{
		float sum = 0.0f;
		for (int k = 0; k < K; ++k) 
		{
			sum += A[row * K + k] * B[k * N + col];
		}
		C[row * N + col] = sum;
	}
}

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

__global__ void matrixMulTiledKernelWithLDG(const float* __restrict__ A,
                                     const float* __restrict__ B,
                                     float* C,
                                     int M, int N, int K)
{
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;

    for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; ++tile)
    {
        int tiledCol = tile * TILE_SIZE + threadIdx.x;
        int tiledRow = tile * TILE_SIZE + threadIdx.y;

        tileA[threadIdx.y][threadIdx.x] = (row < M && tiledCol < K)
            ? __ldg(&A[row * K + tiledCol])
            : 0.0f;

        tileB[threadIdx.y][threadIdx.x] = (col < N && tiledRow < K)
            ? __ldg(&B[tiledRow * N + col])
            : 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k)
        {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N)
    {
        C[row * N + col] = sum;
    }
}

// Each thread computes a 2x2 block of output
__global__ void matrixMulRegisterBlocked(float* A, float* B, float* C, int M, int N, int K) {
    const int BLOCK_SIZE = 2;
    int row = (blockIdx.y * blockDim.y + threadIdx.y) * BLOCK_SIZE;
    int col = (blockIdx.x * blockDim.x + threadIdx.x) * BLOCK_SIZE;
    
    float sum[BLOCK_SIZE][BLOCK_SIZE] = {0};
    
    for (int k = 0; k < K; k++) {
        for (int i = 0; i < BLOCK_SIZE; i++) {
            for (int j = 0; j < BLOCK_SIZE; j++) {
                if (row + i < M && col + j < N) {
                    sum[i][j] += A[(row + i) * K + k] * B[k * N + (col + j)];
                }
            }
        }
    }
    
    for (int i = 0; i < BLOCK_SIZE; i++) {
        for (int j = 0; j < BLOCK_SIZE; j++) {
            if (row + i < M && col + j < N) {
                C[(row + i) * N + (col + j)] = sum[i][j];
            }
        }
    }
}

void matrixMultCPU(float* A, float* B, float* C, int M, int N, int K) 
{
	for (int i = 0; i < M; ++i) 
	{
		for (int j = 0; j < N; ++j) 
		{
			float sum = 0.0f;
			for (int k = 0; k < K; ++k) 
			{
				sum += A[i * K + k] * B[k * N + j];
			}
			C[i * N + j] = sum;
		}
	}
}

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

int main()
{
	try
	{
		constexpr size_t M{1024};
		constexpr size_t K{1024};
		constexpr size_t N{1024};
		constexpr size_t alignment{64};

		// Ensure size is a multiple of alignment
    	auto aligned_size = [](size_t size, size_t alignment) {
    	    return ((size + alignment - 1) / alignment) * alignment;
    	};

		size_t num_bytes_A{M * K * sizeof(float)};
		size_t num_bytes_B{K * N * sizeof(float)};
		size_t num_bytes_C{M * N * sizeof(float)};

		// Host memory with RAII using unique_ptr and free deleter
        auto aligned_alloc_RAII = [alignment](size_t size) {
		    void* ptr = _aligned_malloc(size, alignment);
		    if (!ptr) throw std::bad_alloc();
		    return std::unique_ptr<float[], decltype(&_aligned_free)>{
		        static_cast<float*>(ptr), &_aligned_free
		    };
		};

    	auto A_host = aligned_alloc_RAII(aligned_size(num_bytes_A, alignment));
		auto B_host = aligned_alloc_RAII(aligned_size(num_bytes_B, alignment));
		auto C_host = aligned_alloc_RAII(aligned_size(num_bytes_C, alignment));
		auto C_gpu_result = aligned_alloc_RAII(aligned_size(num_bytes_C, alignment)); // For GPU result

		auto init_matrix = [](float* mat, int rows, int cols, auto func) 
		{
    	for (int i = 0; i < rows; ++i)
    	    for (int j = 0; j < cols; ++j)
    	        mat[i * cols + j] = func(i, j);
		};

		init_matrix(A_host.get(), M, K, [](int i, int j) { return static_cast<float>(i + j); });
		init_matrix(B_host.get(), K, N, [](int i, int j) { return static_cast<float>(i - j); });
		init_matrix(C_host.get(), M, N, [](int, int) { return 0.0f; });

    	CudaBuffer<float> A_device(num_bytes_A);
        CudaBuffer<float> B_device(num_bytes_B);
        CudaBuffer<float> C_device(num_bytes_C);

        CUDA_CHECK(cudaMemcpy(A_device.get(), A_host.get(), num_bytes_A, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(B_device.get(), B_host.get(), num_bytes_B, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(C_device.get(), C_host.get(), num_bytes_C, cudaMemcpyHostToDevice));

		// Performance metrics
		const long long flops = 2LL * M * N * K;
		const long long bytes_accessed = (long long)(M * K + K * N + M * N) * sizeof(float);

		// CPU Timing
		auto startCPU = std::chrono::high_resolution_clock::now();
		matrixMultCPU(A_host.get(), B_host.get(), C_host.get(), M, N, K);
		auto endCPU = std::chrono::high_resolution_clock::now();
		float timeCPU = std::chrono::duration_cast<std::chrono::nanoseconds>(endCPU - startCPU).count() / 1e6f;
		std::cout << "CPU time: " << timeCPU << " ms\n";
			
		// CPU performance summary
		float gflops_cpu = (flops / (timeCPU * 1e-3)) / 1e9f;
		float bandwidth_cpu = (bytes_accessed / (timeCPU * 1e-3)) / 1e9f;
		std::cout << "\nCPU: " << gflops_cpu << " GFLOPS, " << bandwidth_cpu << " GB/s\n";
		// Block and grid sizes
		dim3 blockDim(32, 32);
		dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
		             (M + blockDim.y - 1) / blockDim.y);
			
		// Macro to benchmark a kernel
		#define BENCHMARK_KERNEL(name, kernel_launch) do {                                      \
		    float time = 0.0f;                                                                  \
		    cudaEvent_t start, stop;                                                            \
		    CUDA_CHECK(cudaEventCreate(&start));                                                \
		    CUDA_CHECK(cudaEventCreate(&stop));                                                 \
		    CUDA_CHECK(cudaMemset(C_device.get(), 0, num_bytes_C));                             \
		    CUDA_CHECK(cudaEventRecord(start));                                                 \
		    kernel_launch;                                                                      \
		    CUDA_CHECK(cudaEventRecord(stop));                                                  \
		    CUDA_CHECK(cudaEventSynchronize(stop));                                             \
		    CUDA_CHECK(cudaEventElapsedTime(&time, start, stop));                               \
		    std::cout << #name << " time: " << time << " ms\n";                                 \
		    CUDA_CHECK(cudaMemcpy(C_gpu_result.get(), C_device.get(), num_bytes_C, cudaMemcpyDeviceToHost)); \
		    verifyResult(C_host.get(), C_gpu_result.get(), M * N)                               \
		        ? std::cout << "Results match!\n"                                               \
		        : std::cout << "Results do not match!\n";                                       \
		    float gflops = (flops / (time * 1e-3)) / 1e9f;                                       \
		    float bandwidth = (bytes_accessed / (time * 1e-3)) / 1e9f;                          \
		    std::cout << #name << ": " << gflops << " GFLOPS, " << bandwidth << " GB/s\n\n";    \
		    CUDA_CHECK(cudaEventDestroy(start));                                                \
		    CUDA_CHECK(cudaEventDestroy(stop));                                                 \
		} while (0)
		
		// Launch and measure kernels
		BENCHMARK_KERNEL(Naive,
		   ( matrixMulNaiveKernel<<<gridDim, blockDim>>>(A_device.get(), B_device.get(), C_device.get(), M, N, K)));
		
		BENCHMARK_KERNEL(Tiled,
		    (matrixMulTiledKernel<<<gridDim, blockDim>>>(A_device.get(), B_device.get(), C_device.get(), M, N, K)));
		
		BENCHMARK_KERNEL(Tiled_LDG,
		    (matrixMulTiledKernelWithLDG<<<gridDim, blockDim>>>(A_device.get(), B_device.get(), C_device.get(), M, N, K)));
		
		BENCHMARK_KERNEL(RegisterBlocked,
		    (matrixMulRegisterBlocked<<<gridDim, dim3(16, 16)>>>(A_device.get(), B_device.get(), C_device.get(), M, N, K)));
		
		// cuBLAS benchmark
		cublasHandle_t handle;
		cublasCreate(&handle);
		
		const float alpha = 1.0f;
		const float beta = 0.0f;
		
		cudaEvent_t startCublas, stopCublas;
		CUDA_CHECK(cudaEventCreate(&startCublas));
		CUDA_CHECK(cudaEventCreate(&stopCublas));
		CUDA_CHECK(cudaMemset(C_device.get(), 0, num_bytes_C));
		
		CUDA_CHECK(cudaEventRecord(startCublas));
		cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
		            N, M, K,
		            &alpha,
		            B_device.get(), N,
		            A_device.get(), K,
		            &beta,
		            C_device.get(), N);
		CUDA_CHECK(cudaEventRecord(stopCublas));
		CUDA_CHECK(cudaEventSynchronize(stopCublas));
		
		float timeCublas = 0.0f;
		CUDA_CHECK(cudaEventElapsedTime(&timeCublas, startCublas, stopCublas));
		std::cout << "cuBLAS time: " << timeCublas << " ms\n";
		
		CUDA_CHECK(cudaMemcpy(C_gpu_result.get(), C_device.get(), num_bytes_C, cudaMemcpyDeviceToHost));
		verifyResult(C_host.get(), C_gpu_result.get(), M * N)
		    ? std::cout << "Results match!\n"
		    : std::cout << "Results do not match!\n";
		
		float gflops_cublas = (flops / (timeCublas * 1e-3)) / 1e9f;
		float bandwidth_cublas = (bytes_accessed / (timeCublas * 1e-3)) / 1e9f;
		std::cout << "cuBLAS: " << gflops_cublas << " GFLOPS, " << bandwidth_cublas << " GB/s\n";
		
		CUDA_CHECK(cudaEventDestroy(startCublas));
		CUDA_CHECK(cudaEventDestroy(stopCublas));
		cublasDestroy(handle);
		
		#undef BENCHMARK_KERNEL

	}
	catch (const std::exception& ex) 
	{
        std::cerr << "Unhandled exception: " << ex.what() << std::endl;
        return EXIT_FAILURE;
    }

	
	return EXIT_SUCCESS;
}