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
