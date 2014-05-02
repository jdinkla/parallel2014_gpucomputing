/*
* Copyright (c) 2014 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include "MapBenchmark.h"
#include "CudaTimer.h"
#include "Extent.h"
#include "CudaUtilities.h"
#include "CudaExecConfig.h"
#include "IdFunctor.h"
#include "Map.h"
#include <iostream>
#include "IdFunctor.h"

using namespace std;

template <typename T>
void bench_map_copy_timed(
	CudaTimer& timer,
	Extent& ext,
	thrust::device_vector<T>& d1,
	thrust::device_vector<T>& d2,
	dim3 thr)
{
	timer.start();

	CudaExecConfig cnf(ext, thr);

	IdFunctor<int> op;
	map(cnf, op, ext, d1, d2);

	cudaDeviceSynchronize();
	check_cuda();

	timer.stop();

	cout
		<< thr.x
		<< ";" << timer.delta()
		<< endl;
}

void bench_map_copy()
{
	CudaTimer timer;

	// defs
	const int sizeX = 14 * 1024;
	const int sizeY = 14 * 1024;
	Extent ext(sizeX* sizeY);

	// prepare data
	thrust::device_vector<int> d1(ext.get_number_of_elems());
	thrust::sequence(d1.begin(), d1.end(), 0);
	thrust::device_vector<int> d2(ext.get_number_of_elems());

	cudaDeviceSynchronize();
	check_cuda();
	// 
	for (int t = 1; t <= 1024; t++)
	{
		bench_map_copy_timed(timer, ext, d1, d2, dim3(t, 1, 1));
	}
}

void bench_map_copy_32()
{
	CudaTimer timer;

	// defs
	const int sizeX = 14 * 1024;
	const int sizeY = 14 * 1024;
	Extent ext(sizeX* sizeY);

	// prepare data
	thrust::device_vector<int> d1(ext.get_number_of_elems());
	thrust::sequence(d1.begin(), d1.end(), 0);
	thrust::device_vector<int> d2(ext.get_number_of_elems());

	cudaDeviceSynchronize();
	check_cuda();
	// 
	for (int t = 32; t <= 1024; t += 32)
	{
		bench_map_copy_timed(timer, ext, d1, d2, dim3(t, 1, 1));
	}
}
