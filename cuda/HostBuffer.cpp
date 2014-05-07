/*
* Copyright (c) 2014 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#include <cuda_runtime_api.h>
#include "HostBuffer.h"
#include "Defs.h"

template <typename T>
HostBuffer<T>::HostBuffer(const Extent& e)
: BaseBuffer<T>(e)
{
#ifdef MAC
	const size_t sz = BaseBuffer<T>::extent.get_number_of_elems() * sizeof(T);
	BaseBuffer<T>::ptr = (T*)malloc(sz);
	BaseBuffer<T>::version = 0;
#else
	const size_t sz = extent.get_number_of_elems() * sizeof(T);
	ptr = (T*)malloc(sz);
	version = 0;
#endif	
}

template <typename T>
HostBuffer<T>::~HostBuffer()
{
#ifdef MAC
	if (BaseBuffer<T>::ptr)
	{
		free(BaseBuffer<T>::ptr);
		BaseBuffer<T>::ptr = 0;
		BaseBuffer<T>::version = -1;
	}	
#else
	if (ptr)
	{
		free(ptr);
		ptr = 0;
		version = -1;
	}
#endif
}

// Instances
template HostBuffer<float>;
template HostBuffer<int>;
