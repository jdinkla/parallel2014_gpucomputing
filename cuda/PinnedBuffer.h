/*
* Copyright (c) 2014 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#pragma once

#include "BaseBuffer.h"

template <typename T>
class PinnedBuffer
	: public BaseBuffer<T>
{

public:

	PinnedBuffer(const Extent& e);

	virtual ~PinnedBuffer();

#ifdef CPP11
	// no copies allowed
	PinnedBuffer(PinnedBuffer<T>&) = delete;
#endif

};