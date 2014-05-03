/*
* Copyright (c) 2014 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#include <cuda_runtime_api.h>
#include "BufferPair.h"
#include "CudaUtilities.h"
#include <assert.h>

template <typename T>
BufferPair<T>::BufferPair(PinnedBuffer<T>& _host, DeviceBuffer<T>& _device)
: host(_host)
, device(_device)
{
}

template <typename T>
void BufferPair<T>::update_host()
{
	if (host.get_version() < device.get_version())
	{
		device.copy_to(host);
	}
}

template <typename T>
void BufferPair<T>::update_device()
{
	if (host.get_version() > device.get_version())
	{
		device.copy_from(host);
	}
}

// Instances
template BufferPair<int>;
template BufferPair<float>;
template BufferPair<double>;
