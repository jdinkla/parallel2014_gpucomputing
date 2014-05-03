/*
* Copyright (c) 2014 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#pragma once

#include "PinnedBuffer.h"
#include "DeviceBuffer.h"

template <typename T>
class BufferPair
{

public:

	BufferPair(PinnedBuffer<T>& _host, DeviceBuffer<T>& _device);

	PinnedBuffer<T>& get_host() const
	{
		return host;
	}

	DeviceBuffer<T>& get_device() const
	{
		return device;
	}

	void update_host();

	void update_device();

private:

	PinnedBuffer<T>& host;

	DeviceBuffer<T>& device;

};