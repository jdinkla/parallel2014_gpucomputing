/*
* Copyright (c) 2014 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#pragma once

#include <cuda_runtime_api.h>
#include <iostream>
#include "Extent.h"

/*
	Ein geschachtelter Extent. Zum Zugriff auf ein Element sind vier Koordinaten notwendig

	xo, yo, xi, yo

	o für outer
	i für inner

	Beispiele hierfür sind die Aufteilung eines Images in Grid und Block.
*/

class NestedExtent
{

public:

	__device__ __host__
	NestedExtent(const Extent _outer, const Extent _inner)
		: outer(_outer)
		, inner(_inner)
	{
	}

	__device__ __host__
	int index(const int xo, const int yo, const int xi, const int yi) const
	{
		const int x = xo * inner.get_width() + xi;
		const int y = yo * inner.get_height() + yi;
		return y * get_width() + x;
	}

	__device__ __host__
	int checked_index(const int xo, const int yo, const int xi, const int yi) const
	{
		int result = -1;
		if (in_bounds_strict(xo, yo, xi, yi))
		{
			result = index(xo, yo, xi, yi);
		}
		return result;
	}

	__device__ __host__
	bool in_bounds(const int xo, const int yo, const int xi, const int yi) const
	{
		return outer.in_bounds(xo, yo) && inner.in_bounds(xi, yi);
	}

	__device__ __host__
	bool in_bounds_strict(const int xo, const int yo, const int xi, const int yi) const
	{
		return outer.in_bounds_strict(xo, yo) && inner.in_bounds_strict(xi, yi);
	}

	__device__ __host__
	int get_width() const
	{
		return outer.get_width() * inner.get_width();
	}

	__device__ __host__
	int get_height() const
	{
		return outer.get_height() * inner.get_height();
	}

	__device__ __host__
	int get_number_of_elems() const
	{
		return get_width() * get_height();
	}

	__device__ __host__
	bool operator==(const NestedExtent& b) const
	{
		return this->outer == b.outer && this->inner == b.inner;
	}

	__device__ __host__
	Extent get_outer() const
	{
		return outer;
	}

	__device__ __host__
	Extent get_inner() const
	{
		return inner;
	}

private:

	const Extent outer;
	const Extent inner;

};
