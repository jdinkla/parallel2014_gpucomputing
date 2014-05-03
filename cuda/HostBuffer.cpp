/*
* Copyright (c) 2014 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#include <cuda_runtime_api.h>
#include "HostBuffer.h"

template <typename T>
HostBuffer<T>::HostBuffer(const Extent& e)
: BaseBuffer(e)
{
	const size_t sz = extent.get_number_of_elems() * sizeof(T);
	ptr = (T*)malloc(sz);
	version = 0;
}

template <typename T>
HostBuffer<T>::~HostBuffer()
{
	if (ptr)
	{
		free(ptr);
		ptr = 0;
		version = -1;
	}
}

// Instances
template HostBuffer<float>;
template HostBuffer<int>;
