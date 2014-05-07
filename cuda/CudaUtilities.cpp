#include "CudaUtilities.h"

#include <cuda_runtime_api.h>
#include "CudaException.h"

using namespace std;

void check_cuda()
{
	cudaError_t rc = cudaGetLastError();
	if (rc != cudaSuccess)
	{
		const char* txt = cudaGetErrorString(rc);
		throw CudaException(txt);
	}
}

size_t get_free_device_mem()
{
	size_t free, total;
	cudaError_t rc = cudaMemGetInfo	(&free, &total);
	check_cuda();
	return free;
}

size_t get_total_device_mem()
{
	size_t free, total;
	cudaError_t rc = cudaMemGetInfo	(&free, &total);
	check_cuda();
	return total;
}
