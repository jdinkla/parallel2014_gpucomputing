/*
* Copyright (c) 2014 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#pragma once

#include <cuda_runtime_api.h>
#include "CudaUtilities.h"

struct node
{
	int n;
	node* left;
	node* right;

	node(int _n, node* _l = 0, node* _r = 0)
		: n(_n), left(_l), right(_r)
	{
	}

	void *operator new(size_t len)
	{
		void *ptr;
		cudaMallocManaged(&ptr, len, cudaMemAttachGlobal);
		check_cuda();
		return ptr;
	}

	void operator delete(void *ptr)
	{
		cudaFree(ptr);
		check_cuda();
	}

	__device__ __host__
	bool is_leaf() const
	{
		return left == 0 && right == 0;
	}

};

// CLR90, S. 251
void insert(node** root, node* z);
