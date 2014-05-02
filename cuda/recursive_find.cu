/*
* Copyright (c) 2014 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#include "recursive_find.h"

__global__
void rec_find_kernel(
	const node* n, const int numElems,
	const int* elems, bool* found)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x < numElems)
	{
		const int elem = elems[x];
		found[x] = find(n, elem);
	}
}

void  rec_find(
	CudaExecConfig cnf,
	const node* n,
	thrust::device_vector<int>& elems,
	thrust::device_vector<bool>& found
	)
{
	const size_t numElems = elems.size();
	dim3 g = cnf.get_grid();
	dim3 b = cnf.get_block();
	const int* ePtr = thrust::raw_pointer_cast(&elems[0]);;
	bool* fPr = thrust::raw_pointer_cast(&found[0]);
	rec_find_kernel<<<g, b>>>(n, numElems, ePtr, fPr);
}

