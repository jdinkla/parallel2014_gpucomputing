/*
* Copyright (c) 2014 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#define _CRT_SECURE_NO_WARNINGS 1

#include "while_find.h"

void while_find_host(
	const node* n,
	thrust::host_vector<int>& elems,
	thrust::host_vector<bool>& found,
	const int x)
{
	const size_t numElems = elems.size();
	//const int* ePtr = thrust::raw_pointer_cast(&elems[0]);;
	//bool* fPr = thrust::raw_pointer_cast(&found[0]);
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