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

// Diese Größen ändern, wenn der Speicherplatz nicht ausreicht.
const int sizeX = 14 * 1024;
const int sizeY = 14 * 1024;

template <typename T>
void bench_map_copy_timed(
	CudaTimer& timer,
	Extent& ext,
	thrust::device_vector<T>& d1,
	thrust::device_vector<T>& d2,
	dim3 thr)
{
	timer.start();

	// Rufe Kernel auf und sync
	CudaExecConfig cnf(ext, thr);
	IdFunctor<T> op;
	map(cnf, op, ext, d1, d2);
	cudaDeviceSynchronize();
	check_cuda();

	timer.stop();

	cout
		<< thr.x
		<< ";" << ext.get_number_of_elems() * sizeof(T)
		<< ";" << timer.delta()
		<< endl;
}

template <typename T>
void bench_map(Extent& ext, thrust::device_vector<T>& d1, thrust::device_vector<T>& d2)
{
	CudaTimer timer;
	cudaDeviceSynchronize();
	check_cuda();
	for (int t = 1; t <= 1024; t++)
	{
		bench_map_copy_timed<T>(timer, ext, d1, d2, dim3(t, 1, 1));
	}
}

template <typename T>
void bench_map_32(Extent& ext, thrust::device_vector<T>& d1, thrust::device_vector<T>& d2)
{
	CudaTimer timer;
	cudaDeviceSynchronize();
	check_cuda();
	for (int t = 32; t <= 1024; t += 32)
	{
		bench_map_copy_timed<T>(timer, ext, d1, d2, dim3(t, 1, 1));
	}
}

template <typename T>
void bench_map_32_short(Extent& ext, thrust::device_vector<T>& d1, thrust::device_vector<T>& d2)
{
	CudaTimer timer;
	cudaDeviceSynchronize();
	check_cuda();
	for (int t = 32; t <= 256; t += 32)
	{
		bench_map_copy_timed<T>(timer, ext, d1, d2, dim3(t, 1, 1));
	}
}

void bench_map_copy()
{
	CudaTimer timer;
	Extent ext(sizeX* sizeY);
	thrust::device_vector<int> d1(ext.get_number_of_elems());
	thrust::sequence(d1.begin(), d1.end(), 0);
	thrust::device_vector<int> d2(ext.get_number_of_elems());
	cudaDeviceSynchronize();
	check_cuda();
	bench_map<int>(ext, d1, d2);
}

void bench_map_copy_32()
{
	CudaTimer timer;
	Extent ext(sizeX* sizeY);
	thrust::device_vector<int> d1(ext.get_number_of_elems());
	thrust::sequence(d1.begin(), d1.end(), 0);
	thrust::device_vector<int> d2(ext.get_number_of_elems());
	cudaDeviceSynchronize();
	check_cuda();
	bench_map_32<int>(ext, d1, d2);
}

void bench_map_copy_32_short()
{
	CudaTimer timer;
	Extent ext(sizeX* sizeY);
	thrust::device_vector<int> d1(ext.get_number_of_elems());
	thrust::sequence(d1.begin(), d1.end(), 0);
	thrust::device_vector<int> d2(ext.get_number_of_elems());
	cudaDeviceSynchronize();
	check_cuda();
	bench_map_32_short<int>(ext, d1, d2);
}

void bench_map_copy_int2()
{
	CudaTimer timer;
	Extent ext(sizeX / 2 * sizeY);
	thrust::device_vector<int2> d1(ext.get_number_of_elems());
	thrust::device_vector<int2> d2(ext.get_number_of_elems());
	cudaDeviceSynchronize();
	check_cuda();
	bench_map<int2>(ext, d1, d2);
}

void bench_map_copy_int2_32()
{
	CudaTimer timer;
	Extent ext(sizeX / 2 * sizeY);
	thrust::device_vector<int2> d1(ext.get_number_of_elems());
	thrust::device_vector<int2> d2(ext.get_number_of_elems());
	cudaDeviceSynchronize();
	check_cuda();
	bench_map_32<int2>(ext, d1, d2);
}

void bench_map_copy_int2_32_short()
{
	CudaTimer timer;
	Extent ext(sizeX / 2 * sizeY);
	thrust::device_vector<int2> d1(ext.get_number_of_elems());
	thrust::device_vector<int2> d2(ext.get_number_of_elems());
	cudaDeviceSynchronize();
	check_cuda();
	bench_map_32_short<int2>(ext, d1, d2);
}

void bench_map_copy_int4()
{
	CudaTimer timer;
	Extent ext(sizeX / 4 * sizeY);
	thrust::device_vector<int4> d1(ext.get_number_of_elems());
	thrust::device_vector<int4> d2(ext.get_number_of_elems());
	cudaDeviceSynchronize();
	check_cuda();
	bench_map<int4>(ext, d1, d2);
}

void bench_map_copy_int4_32()
{
	CudaTimer timer;
	Extent ext(sizeX / 4 * sizeY);
	thrust::device_vector<int4> d1(ext.get_number_of_elems());
	thrust::device_vector<int4> d2(ext.get_number_of_elems());
	cudaDeviceSynchronize();
	check_cuda();
	bench_map_32<int4>(ext, d1, d2);
}

void bench_map_copy_int4_32_short()
{
	CudaTimer timer;
	Extent ext(sizeX / 4 * sizeY);
	thrust::device_vector<int4> d1(ext.get_number_of_elems());
	thrust::device_vector<int4> d2(ext.get_number_of_elems());
	cudaDeviceSynchronize();
	check_cuda();
	bench_map_32_short<int4>(ext, d1, d2);
}
