/*
* Copyright (c) 2014 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#include "while_find.h"
#include "CudaExecConfig.h"
#include <thrust/sequence.h>
#include "ThrustUtilities.h"

__global__ 
void while_find_kernel(const node* n, const int numElems,
	const int* elems, bool* found)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x < numElems)
	{
		node* m = const_cast<node*>(n);
		const int elem = elems[x];
		while (m != 0 && m->n != elem)
		{
			if (elem < m->n) m = m->left;
			else m = m->right;
		}
		found[x] = m != 0;
	}
}

void while_find(
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
	while_find_kernel<<<g, b>>>(n, numElems, ePtr, fPr);
}

void cuda_while_example_7()
{
	node* n1 = new node(1);
	node* n3 = new node(3);
	node* n2 = new node(2, n1, n3);
	node* n5 = new node(5);
	node* n7 = new node(7);
	node* n6 = new node(6, n5, n7);
	node* n4 = new node(4, n2, n6);

	thrust::device_vector<int> d(7);
	thrust::sequence(d.begin(), d.end(), 1);
	thrust::device_vector<bool> found(7, false);

	while_find_kernel<<<1, 128>>>(n4, 7, raw_pointer_cast(&d[0]), raw_pointer_cast(&found[0]));

	show(found);
}


