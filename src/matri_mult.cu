#include <cstdlib> 

int main()
{
	auto M{1024};
	auto K{1024};
	auto N{1024};
	auto alignment{64};
    float* A_host = static_cast<float*>(std::aligned_alloc(alignment, M * K * sizeof(float)));
	float* B_host = static_cast<float*>(std::aligned_alloc(alignment, K * N * sizeof(float)));
	float* C_host = static_cast<float*>(std::aligned_alloc(alignment, M * N * sizeof(float)));

	for(int i = 0; i < M; i++) 
	{
    	for(int j = 0; j < K; j++) 
		{
    	    A_host[i * K + j] = static_cast<float>(i + j);
    	}
	}

	for(int i = 0; i < K; i++) 
	{
    	for(int j = 0; j < N; j++) 
		{
    	    B_host[i * N + j] = static_cast<float>(i - j);
    	}
	}
	for(int i = 0; i < M; i++) 
	{
		for(int j = 0; j < N; j++) 
		{
		    C_host[i * N + j] = 0.0f;
		}
	}

	std::free(A_host);
    std::free(B_host);
    std::free(C_host);	
}