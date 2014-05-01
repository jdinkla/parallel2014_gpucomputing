/*
* Copyright (c) 2014 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#pragma once

#include <cuda_runtime_api.h>
#include "Extent.h"

template <typename T>
class BaseBuffer
{

public:

	BaseBuffer(const Extent& e)
		: ptr(0)
		, extent(e)
	{
	}

	virtual ~BaseBuffer()
	{
	}

	__device__ __host__
	virtual T* get_ptr() const
	{
		return ptr;
	}

	__device__ __host__
	virtual Extent get_extent() const
	{
		return extent;
	}

    virtual size_t get_size_in_bytes() const
    {
        return extent.get_number_of_elems() * sizeof(T);
    }
    
    T& operator[](const int idx) const {
        return ptr[idx];
    }

    T* begin() const
    {
        return ptr;
    }
    
    T* end() const
    {
        return ptr + extent.get_number_of_elems();
    }
    
protected:

	Extent extent;

	T* ptr;

};