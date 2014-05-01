/*
* Copyright (c) 2014 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#pragma once

#include <cuda_runtime_api.h>

template <typename T>
class AddFunctor
{
public:

	AddFunctor(const T _n) : n(_n) {}

	__host__ __device__
	virtual T operator()(const T value) const
	{
		return value + n;
	}

private:

	const T n;

};



