/*
* Copyright (c) 2014 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#include "Rectangle.h"
#include "Mandelbrot.h"
#include "Extent.h"
#include "CudaUtilities.h"

using thrust::raw_pointer_cast;
using namespace std;

__global__
void mandelbrot_kernel(Extent ext, int* dest, const int max_iter, Rectangle r)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int i = ext.checked_index(x, y);
	if (i >= 0)
	{
		const float x0 = scale(x, ext.get_width(), r.x0, r.x1);
		const float y0 = scale(y, ext.get_height(), r.y0, r.y1);
		dest[i] = mandelbrot(x0, y0, max_iter);
	}
}

void cuda_mandelbrot(
	CudaExecConfig& cnf,
	Extent& ext,
	thrust::device_vector<int>& d,
	const int max_iter,
	Rectangle r)
{
	dim3 g = cnf.get_grid();
	dim3 b = cnf.get_block();
	mandelbrot_kernel<<<g, b>>>(ext, thrust::raw_pointer_cast(&d[0]), max_iter, r);
}


void cuda_mandelbrot()
{
	CudaTimer timer;
	const int max_iter = 1000;
	Rectangle r(-2.5, 1.0, -1.0, 1.0);
	const int s = 1024 * 10;
	Extent ext(s, s);
	CudaExecConfig cnf(ext, dim3(32, 4, 1));
	dim3 g = cnf.get_grid();
	dim3 b = cnf.get_block();
	thrust::device_vector<int> d(ext.get_number_of_elems());
	cudaDeviceSynchronize();
	timer.start();
	mandelbrot_kernel << <g, b >> >(ext, raw_pointer_cast(&d[0]), max_iter, r);
	cudaDeviceSynchronize();
	timer.stop();
	check_cuda();
	cout << "Mandelbrot: " << timer.delta() << endl;
}


void bench_mandelbrot_single(
	CudaTimer& timer,
	Extent& ext,
	thrust::device_vector<int>& d,
	dim3 thr,
	const int max_iter,
	Rectangle r
	)
{
	timer.start();

	CudaExecConfig cnf(ext, thr);
	dim3 g = cnf.get_grid();
	dim3 b = cnf.get_block();
	mandelbrot_kernel << <g, b >> >(ext, raw_pointer_cast(&d[0]), max_iter, r);

	cudaDeviceSynchronize();
	check_cuda();

	timer.stop();

	cout
		<< thr.x << ", " << thr.y << ";"
		<< max_iter << ";"
		<< timer.delta()
		<< endl;
}