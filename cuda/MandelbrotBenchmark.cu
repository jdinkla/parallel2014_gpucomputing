/*
* Copyright (c) 2014 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#include "MandelbrotBenchmark.h"
#include <iostream>
#include "CudaExecConfig.h"
#include "CudaUtilities.h"
#include "Mandelbrot.h"
#include "BenchmarkUtilities.h"
#include "Timer.h"
#include "MandelbrotCPU.h"

using namespace std;

std::vector<int> get_max_iters()
{
	std::vector<int> result;
	result.push_back(1);
	result.push_back(2);
	result.push_back(3);
	result.push_back(4);
	result.push_back(5);
	result.push_back(6);
	result.push_back(7);
	result.push_back(8);
	result.push_back(9);
	result.push_back(10);
	result.push_back(20);
	result.push_back(30);
	result.push_back(40);
	result.push_back(50);
	result.push_back(100);
	result.push_back(1000);
	result.push_back(10000);
	return result;
}


void bench_mandelbrot()
{
	const int sizeX = 14 * 1024;
	const int sizeY = 14 * 1024;
	CudaTimer timer;

	Extent ext(sizeX, sizeY);
	Rectangle r(-2.5, 1.0, -1.0, 1.0);

	// prepare data
	thrust::device_vector<int> d(ext.get_number_of_elems());

	cudaDeviceSynchronize();
	check_cuda();
	// 
	pairs2d blocks = get_blocks();
	std::vector<int> maxis = get_max_iters();

	for (std::vector<int>::iterator it1 = maxis.begin(); it1 != maxis.end(); it1++)
	{
		const int maxi = *it1;
		for (pairs2d::iterator it = blocks.begin(); it != blocks.end(); it++)
		{
			pair2d xy = *it;
			bench_mandelbrot_single(timer, ext, d, dim3(xy.first, xy.second, 1), maxi, r);
		}
	}

}

void bench_mandelbrot_short()
{
	const int sizeX = 14 * 1024;
	const int sizeY = 14 * 1024;
	CudaTimer timer;

	Extent ext(sizeX, sizeY);
	Rectangle r(-2.5, 1.0, -1.0, 1.0);

	// prepare data
	thrust::device_vector<int> d(ext.get_number_of_elems());

	cudaDeviceSynchronize();
	check_cuda();
	// 
	pairs2d blocks = get_blocks();
	std::vector<int> maxis = get_max_iters();

	// Warm up
	bench_mandelbrot_single(timer, ext, d, dim3(192, 1, 1), 10, r);

	const int maxi = 1000;
	for (pairs2d::iterator it = blocks.begin(); it != blocks.end(); it++)
	{
		pair2d xy = *it;
		bench_mandelbrot_single(timer, ext, d, dim3(xy.first, xy.second, 1), maxi, r);
	}

}

void bench_mandelbrot_shorter()
{
	const int sizeX = 14 * 1024;
	const int sizeY = 14 * 1024;
	CudaTimer timer;

	Extent ext(sizeX, sizeY);
	Rectangle r(-2.5, 1.0, -1.0, 1.0);

	// prepare data
	thrust::device_vector<int> d(ext.get_number_of_elems());

	cudaDeviceSynchronize();
	check_cuda();
	// 
	pairs2d blocks = get_special_blocks();
	std::vector<int> maxis = get_max_iters();

	const int maxi = 10000;
	for (pairs2d::iterator it = blocks.begin(); it != blocks.end(); it++)
	{
		pair2d xy = *it;
		bench_mandelbrot_single(timer, ext, d, dim3(xy.first, xy.second, 1), maxi, r);
	}

}

void bench_mandelbrot_128()
{
	const int sizeX = 1 * 1024;
	const int sizeY = 1 * 1024;
	CudaTimer timer, timer2;

	Extent ext(sizeX, sizeY);
	Rectangle r(-2.5, 1.0, -1.0, 1.0);

	// prepare data
	thrust::host_vector<int> h(ext.get_number_of_elems());
	thrust::device_vector<int> d(ext.get_number_of_elems());
	cudaDeviceSynchronize();
	check_cuda();
	// 
	pairs2d blocks;
	blocks.push_back(pair2d(32, 4));
	const int maxi = 1000;
	cout << "size=" << ext.get_number_of_elems() << endl
		<< "max_iter=" << maxi << endl;
	for (pairs2d::iterator it = blocks.begin(); it != blocks.end(); it++)
	{
		pair2d xy = *it;
		timer2.start();
		bench_mandelbrot_single(timer, ext, d, dim3(xy.first, xy.second, 1), maxi, r);
		timer2.stop();
		cout << "GPU: " << timer2.delta() << endl;

		//timer2.start();
		//bench_mandelbrot_single_cpu(timer, ext, h, maxi, r);
		//timer2.stop();
		//cout << "CPU: " << timer2.delta() << endl;

	}
}

void bench_mandelbrot_128_compare()
{
	const int sizeX = 1 * 1024;
	const int sizeY = 1 * 1024;
	CudaTimer timer;
	Timer timer2;

	Extent ext(sizeX, sizeY);
	Rectangle r(-2.5, 1.0, -1.0, 1.0);

	// prepare data
	thrust::host_vector<int> h1(ext.get_number_of_elems());
	thrust::host_vector<int> h2(ext.get_number_of_elems());
	thrust::device_vector<int> d(ext.get_number_of_elems());
	cudaDeviceSynchronize();
	check_cuda();
	// 
	pairs2d blocks;
	blocks.push_back(pair2d(32, 4));
	const int maxi = 1000;
	cout << "size=" << ext.get_number_of_elems() << endl
		<< "max_iter=" << maxi << endl;
	for (pairs2d::iterator it = blocks.begin(); it != blocks.end(); it++)
	{
		pair2d xy = *it;
		timer2.start();
		bench_mandelbrot_single(timer, ext, d, dim3(xy.first, xy.second, 1), maxi, r);
		timer2.stop();
		cout << "GPU:     " << timer2.delta() << endl;

		timer2.start();
		bench_mandelbrot_single_cpu_seq(timer, ext, h1, maxi, r);
		timer2.stop();
		cout << "CPU Seq: " << timer2.delta() << endl;

		for (int i = 0; i < ext.get_number_of_elems(); i++)
		{
			if (d[i] != h1[i])
			{
				cerr << "ERROR " << i << ": " << d[i] << " vs. " << h1[i] << endl;
			}
		}
		timer2.start();
		bench_mandelbrot_single_cpu_par(timer, ext, h2, maxi, r);
		timer2.stop();
		cout << "CPU Par: " << timer2.delta() << endl;

	}
}
