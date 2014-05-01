/*
* Copyright (c) 2014 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#include "node.h"

// CLR90, S. 251
void insert(node** root, node* z)
{
	node* y = 0;
	node* x = *root;
	while (x != 0)
	{
		y = x;
		if (z->n < x->n) x = x->left;
		else x = x->right;
	}
	if (y == 0) *root = z;
	else if (z->n < y->n) y->left = z;
	else if (z->n > y->n) y->right = z; // gleiche elemente nicht einfügen
}
