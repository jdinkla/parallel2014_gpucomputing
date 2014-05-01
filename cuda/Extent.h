/*
 * Copyright (c) 2014 by Joern Dinkla, www.dinkla.com, All rights reserved.
 *
 * See the LICENSE file in the root directory.
 */

#pragma once

#include <cuda_runtime_api.h>
#include <iostream>

class Extent
{

public:

	__device__ __host__
	Extent(const int _width = 1, const int _height = 1)
		: width(_width)
		, height(_height)
	{
	}

	__device__ __host__
	Extent(const Extent& extent)
		: width(extent.width)
		, height(extent.height)
	{
	}

	__device__ __host__
	int index(const int x, const int y) const
	{
		return y * width + x;
	}

	__device__ __host__
	int checked_index(const int x, const int y = 0) const
	{
		int result = -1;
		if (0 <= x && x < width && 0 <= y && y < height)
		{
			result = y * width + x;
		}
		return result;
	}

	__device__ __host__
	bool in_bounds(const int x, const int y = 0) const
	{
		return x < width && y < height;
	}

	__device__ __host__
	bool in_bounds_strict(const int x, const int y = 0) const
	{
		return 0 <= x && x < width && 0 <= y && y < height;
	}

	__device__ __host__
	int get_width() const
	{
		return width;
	}

	__device__ __host__
	int get_height() const
	{
		return height;
	}

	__device__ __host__
	int get_number_of_elems() const
	{
		return width * height;
	}

	__device__ __host__
	Extent& operator=(const Extent& a)
	{
		width = a.width;
		height = a.height;
		return *this;
	}

	__device__ __host__
	bool operator==(const Extent& b) const
	{
		return this->width == b.width && this->height == b.height;
	}

private:

	int width;
	int height;

};

inline std::ostream &operator<<(std::ostream& ostr, const Extent& d)
{
	return ostr << d.get_width() << "," << d.get_height();
}
