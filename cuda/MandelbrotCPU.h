/*
* Copyright (c) 2014 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#include <cuda_runtime_api.h>
#include "Extent.h"
#include "Rectangle.h"
#include "ITimer.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "CudaExecConfig.h"

void mandelbrot_cpu_seq(Extent ext, int* dest, const int max_iter, Rectangle mi);

void mandelbrot_cpu_par(Extent ext, int* dest, const int max_iter, Rectangle mi);

void bench_mandelbrot_single_cpu_seq(
	ITimer& timer,
	Extent& ext,
	thrust::host_vector<int>& h,
	const int max_iter,
	Rectangle r
	);

void bench_mandelbrot_single_cpu_par(
	ITimer& timer,
	Extent& ext,
	thrust::host_vector<int>& h,
	const int max_iter,
	Rectangle r
	);
