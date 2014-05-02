/*
* Copyright (c) 2014 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#include "node.h"
#include <stdio.h>

#if (__CUDA_ARCH__ >= 350)

// Rekursive Kernel werden erst ab CC 3.5 unterstützt.

__global__ void rec_kernel(const node* n)
{
	if (n == 0) return;
	if (n->is_leaf())
	{
		printf("Leaf %d\n", n->n);
	}
	else
	{
		printf("Node %d\n", n->n);
		rec_kernel<<<1, 1>>>(n->left);
		rec_kernel<<<1, 1>>>(n->right);
	}
}

#else

__global__ void rec_kernel(const node* n)
{
	// empty	
}

#endif

void cuda_recursive_kernel()
{
	node* n1 = new node(1);
	node* n3 = new node(3);
	node* n2 = new node(2, n1, n3);

	node* n5 = new node(5);
	node* n7 = new node(7);
	node* n6 = new node(6, n5, n7);

	node* n4 = new node(4, n2, n6);

	rec_kernel<<<1, 1>>>(n4);
	cudaDeviceSynchronize();
}



