/*
* Copyright (c) 2014 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#include <cuda_runtime_api.h>
#include "PinnedBuffer.h"
#include "CudaUtilities.h"

template <typename T>
PinnedBuffer<T>::PinnedBuffer(const Extent& e)
: BaseBuffer(e)
{
	const size_t sz = extent.get_number_of_elems() * sizeof(T);
	cudaMallocHost((void**)&ptr, sz);
	check_cuda();
}

template <typename T>
PinnedBuffer<T>::~PinnedBuffer()
{
	if (ptr)
	{
		cudaFreeHost(ptr);
		ptr = 0;
		check_cuda();
	}
}

// Instances
template PinnedBuffer<float>;
template PinnedBuffer<int>;
