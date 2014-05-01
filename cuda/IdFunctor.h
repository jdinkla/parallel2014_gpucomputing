/*
* Copyright (c) 2014 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#pragma once

#include <cuda_runtime_api.h>

template <typename T>
struct IdFunctor
{
	__host__ __device__
	T operator()(const T v) const
	{
		return v;
	}
};
