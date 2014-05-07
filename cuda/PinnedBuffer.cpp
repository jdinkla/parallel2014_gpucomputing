/*
* Copyright (c) 2014 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#include <cuda_runtime_api.h>
#include "PinnedBuffer.h"
#include "CudaUtilities.h"
#include "Defs.h"

template <typename T>
PinnedBuffer<T>::PinnedBuffer(const Extent& e)
: BaseBuffer<T>(e)
{
#ifdef MAC
	const size_t sz = BaseBuffer<T>::extent.get_number_of_elems() * sizeof(T);
	cudaMallocHost((void**)&BaseBuffer<T>::ptr, sz);
	check_cuda();
	BaseBuffer<T>::version = 0;
#else	
	const size_t sz = extent.get_number_of_elems() * sizeof(T);
	cudaMallocHost((void**)&ptr, sz);
	check_cuda();
	version = 0;
#endif	
}

template <typename T>
PinnedBuffer<T>::~PinnedBuffer()
{
#ifdef MAC
	if (BaseBuffer<T>::ptr)
	{
		cudaFreeHost(BaseBuffer<T>::ptr);
		BaseBuffer<T>::ptr = 0;
		check_cuda();
		BaseBuffer<T>::version = -1;
	}
#else	
	if (ptr)
	{
		cudaFreeHost(ptr);
		ptr = 0;
		check_cuda();
		version = -1;
	}
#endif	
}

// Instances
template PinnedBuffer<float>;
template PinnedBuffer<int>;
