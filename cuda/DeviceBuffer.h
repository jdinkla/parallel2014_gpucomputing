/*
* Copyright (c) 2014 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#pragma once

#include "BaseBuffer.h"
#include "PinnedBuffer.h"

template <typename T>
class DeviceBuffer
	: public BaseBuffer<T>
{

public:

	DeviceBuffer(const Extent& e);

	virtual ~DeviceBuffer();

	void copy_from(const PinnedBuffer<T>& buf);

	void copy_to(PinnedBuffer<T>& buf) const;

#ifdef CPP11
	// no copies allowed
	DeviceBuffer(const DeviceBuffer<T>&) = delete;
#endif

};