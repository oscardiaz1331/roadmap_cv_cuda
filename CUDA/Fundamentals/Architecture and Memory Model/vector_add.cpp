#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

int main() {
    // 1. Vector size (2^20)
    const std::size_t vector_size = static_cast<std::size_t>(std::pow(2.0, 20.0));

    // 2. Host vectors (RAII-managed)
    std::vector<float> A_host(vector_size, 1.0f);
    std::vector<float> B_host(vector_size, 2.0f);
    std::vector<float> C_host(vector_size, 0.0f);

    // 3. Time the vector addition on the CPU
    auto start = std::chrono::high_resolution_clock::now();

    std::transform(A_host.begin(), A_host.end(),
                   B_host.begin(),
                   C_host.begin(),
                   std::plus{});

    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = stop - start;
    std::printf("Elapsed time: %.6f ms\n", elapsed.count());

    return 0;
}
