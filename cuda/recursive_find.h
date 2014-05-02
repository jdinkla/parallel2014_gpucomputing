/*
* Copyright (c) 2014 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#pragma once

#include <cuda_runtime_api.h>
#include "node.h"
#include "CudaExecConfig.h"
#include <thrust/device_vector.h>

__host__ __device__ inline
bool find(const node* n, const int elem)
{
	if (n == 0)				return false;
	if (n->n == elem)		return true;
	else if (elem < n->n)	return find(n->left, elem);
	else					return find(n->right, elem);
}

void rec_find(
	CudaExecConfig cnf,
	const node* n,
	thrust::device_vector<int>& elems,
	thrust::device_vector<bool>& found
	);

