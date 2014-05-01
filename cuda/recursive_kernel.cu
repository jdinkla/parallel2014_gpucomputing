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
}

#endif
