/*
 * Copyright (c) 2014 by Joern Dinkla, www.dinkla.com, All rights reserved.
 *
 * See the LICENSE file in the root directory.
 */

#pragma once

#include <cuda_runtime_api.h>
#include "Extent.h"

class CudaExecConfig
{

public:
	
	CudaExecConfig(const Extent& extent, const dim3 _block, const int _shared_mem = 0, cudaStream_t _stream = 0);

	dim3 get_grid() const
	{
		return grid;
	}

	dim3 get_block() const
	{
		return block;
	}

	int get_shared_mem() const
	{
		return shared_mem;
	}

	cudaStream_t get_stream() const
	{
		return stream;
	}

private:

	dim3 grid;
	dim3 block;
	int shared_mem;
	cudaStream_t stream;

};