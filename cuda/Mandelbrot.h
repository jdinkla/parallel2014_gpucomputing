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

__device__ __host__ inline
float scale(const int c, const int s, const float min, const float max)
{
	const float r = float(c) / float(s);			// 0 <= rel <= 1
	const float size = max - min;
	return min + r * size;
}

__device__ __host__ inline
int mandelbrot(const float x0, const float y0, const int max_iter)
{
	int iter = 0;
	float x = 0.0f;
	float y = 0.0f;
	while (x*x + y*y < 2 * 2 && iter < max_iter)
	{
		const float tmp = x*x - y*y + x0;
		y = 2 * x*y + y0;
		x = tmp;
		++iter;
	}
	return iter;
}

void cuda_mandelbrot(
	CudaExecConfig& cnf,
	Extent& ext,
	thrust::device_vector<int>& d,
	const int max_iter,
	Rectangle r);

void cuda_mandelbrot();

void bench_mandelbrot_single(
	CudaTimer& timer,
	Extent& ext,
	thrust::device_vector<int>& d,
	dim3 thr,
	const int max_iter,
	Rectangle r
	);
