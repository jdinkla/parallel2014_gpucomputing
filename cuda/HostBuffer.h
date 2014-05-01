/*
* Copyright (c) 2014 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#pragma once

#include "BaseBuffer.h"

template <typename T>
class HostBuffer
	: public BaseBuffer<T>
{

public:

	HostBuffer(const Extent& e);

	virtual ~HostBuffer();

	// TODO register, unregister

};