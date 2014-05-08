/*
* Copyright (c) 2014 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#include <cuda_runtime_api.h>
#include "CudaExecConfig.h"
#include "Rectangle.h"
#include <thrust/device_vector.h>
#include "CudaTimer.h"
#include "Extent.h"
#include <vector_functions.h>

__device__ __host__ inline
float2 operator +(float2 a, float2 b) 
{
	return make_float2(a.x + b.x, a.y + b.y);
}

__device__ __host__ inline
float square(const float2 f) 
{
	return f.x * f.x + f.y * f.y;
}
__device__ __host__ inline
float2 iterate(const float2 f) 
{
	return make_float2(f.x * f.x - f.y *f.y, 2 * f.x * f.y);
}

__device__ __host__ inline
int mandelbrot2(const float2 xy0, const int max_iter)
{
	int iter = 0;
	float2 xy = make_float2(0.0f, 0.0f);
	while (square(xy) < 4 && iter < max_iter)
	{
		xy = iterate(xy) + xy0;
		++iter;
	}
	return iter;
}

void cuda_mandelbrot2(
	CudaExecConfig& cnf,
	Extent& ext,
	thrust::device_vector<int>& d,
	const int max_iter,
	Rectangle r);

void cuda_mandelbrot2();

void bench_mandelbrot2_single(
	CudaTimer& timer,
	Extent& ext,
	thrust::device_vector<int>& d,
	dim3 thr,
	const int max_iter,
	Rectangle r
	);
