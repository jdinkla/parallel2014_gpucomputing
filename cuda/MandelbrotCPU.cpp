/*
* Copyright (c) 2014 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#define _CRT_SECURE_NO_WARNINGS 1

#include "Defs.h"
#ifndef MAC
#include <omp.h>
#endif

#include "MandelbrotCPU.h"
#include "Extent.h"
#include "Rectangle.h"
#include "Mandelbrot.h"
#include "ITimer.h"

#include <iostream>
#include <thrust/host_vector.h>

using namespace std;

void mandelbrot_cpu_seq(Extent ext, int* dest, const int max_iter, Rectangle mi)
{
	for (int y = 0; y < ext.get_height(); y++)
	{
		for (int x = 0; x < ext.get_width(); x++)
		{
			const int i = ext.checked_index(x, y);
			const float x0 = scale(x, ext.get_width(), mi.x0, mi.x1);
			const float y0 = scale(y, ext.get_height(), mi.y0, mi.y1);
			dest[i] = mandelbrot(x0, y0, max_iter);
		}
	}
}

void mandelbrot_cpu_par(Extent ext, int* dest, const int max_iter, Rectangle mi)
{
#pragma omp parallel for
	for (int y = 0; y < ext.get_height(); y++)
	{
		for (int x = 0; x < ext.get_width(); x++)
		{
			const int i = ext.checked_index(x, y);
			const float x0 = scale(x, ext.get_width(), mi.x0, mi.x1);
			const float y0 = scale(y, ext.get_height(), mi.y0, mi.y1);
			dest[i] = mandelbrot(x0, y0, max_iter);
		}
	}
}

void bench_mandelbrot_single_cpu_seq(
	ITimer& timer,
	Extent& ext,
	thrust::host_vector<int>& h,
	const int max_iter,
	Rectangle r
	)
{
	timer.start();
	mandelbrot_cpu_seq(ext, thrust::raw_pointer_cast(&h[0]), max_iter, r);
	timer.stop();
	cout
		<< max_iter << ";"
		<< timer.delta()
		<< endl;
}

void bench_mandelbrot_single_cpu_par(
	ITimer& timer,
	Extent& ext,
	thrust::host_vector<int>& h,
	const int max_iter,
	Rectangle r
	)
{
	timer.start();
	mandelbrot_cpu_par(ext, thrust::raw_pointer_cast(&h[0]), max_iter, r);
	timer.stop();
	cout
		<< max_iter << ";"
		<< timer.delta()
		<< endl;
}
