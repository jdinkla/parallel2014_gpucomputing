/*
* Copyright (c) 2014 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#include <cuda_runtime_api.h>
#include "DeviceBuffer.h"
#include "CudaUtilities.h"
#include <assert.h>
#include "Defs.h"

template <typename T>
DeviceBuffer<T>::DeviceBuffer(const Extent& e)
: BaseBuffer<T>(e)
{
#ifdef MAC
	size_t sz = BaseBuffer<T>::extent.get_number_of_elems() * sizeof(T);
	cudaMalloc((void**)&BaseBuffer<T>::ptr, sz);
	check_cuda();
	BaseBuffer<T>::version = 0;
#else
	size_t sz = BaseBuffer<T>::extent.get_number_of_elems() * sizeof(T);
	cudaMalloc((void**)&ptr, sz);
	check_cuda();
	version = 0;
#endif	
}

template <typename T>
DeviceBuffer<T>::~DeviceBuffer()
{
	if (BaseBuffer<T>::ptr)
	{
		cudaFree(BaseBuffer<T>::ptr);
		check_cuda();
#ifdef MAC
		BaseBuffer<T>::version = -1;
#else
		version = -1;
#endif		
	}
}

template <typename T>
void DeviceBuffer<T>::copy_from(const PinnedBuffer<T>& buf)
{
	assert(buf.get_extent() == BaseBuffer<T>::get_extent());
	size_t sz = BaseBuffer<T>::get_extent().get_number_of_elems() * sizeof(T);
	void* dst = BaseBuffer<T>::get_ptr();
	void* src = buf.get_ptr();
	cudaMemcpy(dst, src, sz, cudaMemcpyHostToDevice);
	check_cuda();
	set_version(buf);
}

template <typename T>
void DeviceBuffer<T>::copy_to(PinnedBuffer<T>& buf) const
{
	assert(buf.get_extent() == BaseBuffer<T>::get_extent());
	size_t sz = BaseBuffer<T>::get_extent().get_number_of_elems() * sizeof(T);
	void* dst = buf.get_ptr();
	void* src = BaseBuffer<T>::get_ptr();
	cudaMemcpy(dst, src, sz, cudaMemcpyDeviceToHost);
	check_cuda();
	buf.set_version(*this);
}

// Instances
template DeviceBuffer<int>;
template DeviceBuffer<float>;
template DeviceBuffer<double>;
