/*
* Copyright (c) 2014 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#pragma once

#include "CudaExecConfig.h"
#include "Extent.h"
#include <thrust/device_vector.h>

template <class F, typename I, typename O>
__global__
void map_kernel(const F op, Extent ext, const I* src, O* dest)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int i = ext.checked_index(x);
	if (i >= 0)
	{
		dest[i] = op(src[i]);
	}
}

template <class F, typename I, typename O>
void map(CudaExecConfig& cnf, const F& op, Extent& ext,
	thrust::device_vector<I>& src, thrust::device_vector<O>& dest);

template <class F, typename I, typename O>
__global__ 
void map_2d_kernel(const F op, Extent ext, const I* src, O* dest)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int i = ext.checked_index(x, y);
	if (i >= 0)
	{
		dest[i] = op(src[i]);
	}
}

template <class F, typename I, typename O>
void map_2d(CudaExecConfig& cnf, const F& op, Extent& ext,
	thrust::device_vector<I>& src, thrust::device_vector<O>& dest);

template <class F, typename I, typename O>
__global__ 
void map_ext_2d_kernel(const F op, Extent ext, const I* src, O* dest)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int i = ext.checked_index(x, y);
	if (i >= 0)
	{
		dest[i] = op(src, i, x, y);
	}
}


