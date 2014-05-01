/*
* Copyright (c) 2014 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#include "recursive_find.h"

__global__
void rec_find(
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
