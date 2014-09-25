/*
* Copyright (c) 2014 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#include <algorithm>				// CUDA 6.5 requires this for std::min

#include "Defs.h"
#ifndef MAC
#include <omp.h>
#endif
#include <cuda_runtime_api.h>
#include "CudaUtilities.h"
#include <vector>
#include "CudaExecConfig.h"
#include "Partitions.h"

__global__ void kernel(int* src, int* dst)
{

}

void call_kernel(CudaExecConfig& cnf, cudaStream_t& stream, int* src, int* dst)
{
	dim3 grd = cnf.get_grid();
	dim3 blk = cnf.get_block();
	kernel<<<grd, blk, 0, stream>>>(src, dst);
}

void cuda_multi_gpu()
{
	dim3 grd(1, 1, 1);
	dim3 blk(1, 1, 1);
	Extent ext(1024, 1024);
	CudaExecConfig cnf(ext, dim3(128, 1, 1));
	const int numDevices = 4;
	const int numStreamsPerDevice = 2;
	const int numParts = numDevices * numStreamsPerDevice;
	const int numJobs = 1024;

	size_t sz = 1024;

	int* src_d[8];
	int* src_h[8];
	int* dest_h[8];
	int* dest_d[8];

	// Partitions
	Partitions parts(numJobs, numParts);
	// Konfiguriere OpenMP
#ifndef MAC
	const int old_threads = omp_get_num_threads();
	omp_set_num_threads(numParts);
#endif
#pragma omp parallel for
	for (int i = 0; i < numParts; i++)
	{
		Partition* p = parts.get(i);
		const int id = p->partition_id;
		cudaSetDevice(p->device_id);															check_cuda();
		// fork, d.h. in Queue einsetzen
		for (int j = p->start; j <= p->end; j++)
		{
			cudaMemcpyAsync(src_d[id], src_h[id], sz, cudaMemcpyHostToDevice, p->stream);
			call_kernel(cnf, p->stream, src_d[id], dest_d[id]);
			// geht nicht wg. OpenMP: kernel<<<grd, blk, 0, stream >> >(src, dst);			
			cudaMemcpyAsync(dest_h[id], dest_d[id], sz, cudaMemcpyDeviceToHost, p->stream);
		}
		// Und join
		cudaStreamSynchronize(p->stream);														check_cuda();
	}
#ifndef MAC	
	omp_set_num_threads(old_threads);
#endif	
}