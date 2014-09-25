/*
* Copyright (c) 2014 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#include <algorithm>				// CUDA 6.5 requires this for std::min
#include "Map.h"
#include "CudaExecConfig.h"
#include "Extent.h"
#include <thrust/device_vector.h>
#include "IdFunctor.h"

template <class F, typename I, typename O>
void map(CudaExecConfig& cnf, const F& op, Extent& ext,
	thrust::device_vector<I>& src, thrust::device_vector<O>& dest)
{
	const I* srcPtr = thrust::raw_pointer_cast(&src[0]);
	O* destPtr = thrust::raw_pointer_cast(&dest[0]);
	map_kernel<<<cnf.get_grid(), cnf.get_block()>>>(op, ext, srcPtr, destPtr);
}

template <class F, typename I, typename O>
void map_2d(CudaExecConfig& cnf, const F& op, Extent& ext, 
	thrust::device_vector<I>& src, thrust::device_vector<O>& dest)
{
	const I* srcPtr = thrust::raw_pointer_cast(&src[0]);
	O* destPtr = thrust::raw_pointer_cast(&dest[0]);
	map_2d_kernel<<<cnf.get_grid(), cnf.get_block()>>>(op, ext, srcPtr, destPtr);
}

// Instances

template
void map(CudaExecConfig& cnf, const IdFunctor<int>& op, Extent& ext,
thrust::device_vector<int>& src, thrust::device_vector<int>& dest);

template
void map(CudaExecConfig& cnf, const IdFunctor<int2>& op, Extent& ext,
thrust::device_vector<int2>& src, thrust::device_vector<int2>& dest);

template
void map(CudaExecConfig& cnf, const IdFunctor<int4>& op, Extent& ext,
thrust::device_vector<int4>& src, thrust::device_vector<int4>& dest);

