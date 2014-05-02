/*
* Copyright (c) 2014 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#pragma once

#include <cuda_runtime_api.h>
#include "CudaTimer.h"
#include "Extent.h"
#include <thrust/device_vector.h>
#include "Rectangle.h"
#include <vector>

void bench_mandelbrot_single(
	CudaTimer& timer,
	Extent& ext,
	thrust::device_vector<int>& d,
	dim3 thr,
	const int max_iter,
	Rectangle r
	);

void bench_mandelbrot();

void bench_mandelbrot_short();

void bench_mandelbrot_shorter();

void bench_mandelbrot_128();

void bench_mandelbrot_128_compare();

std::vector<int> get_max_iters();

