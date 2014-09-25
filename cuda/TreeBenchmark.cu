/*
* Copyright (c) 2014 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#include <algorithm>				// CUDA 6.5 requires this for std::min
#include "Defs.h"
#include "TreeBenchmark.h"
#ifndef MAC
#include <random>
#else
#include <stdlib.h>
#endif
#include "node.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "Extent.h"
#include "CudaExecConfig.h"
#include "while_find.h"
#include "recursive_find.h"

void tree_compare_rec_while()
{
	const int treeSize = 8 * 1024;
	const int searchSize = 1 * 1024;

	const int low = 0;
	const int high = 2 * treeSize;

	// Erstelle Baum und Sucharray
	node* root = 0;
	thrust::host_vector<int> h(searchSize);

#ifndef MAC
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<> dist(low, high);
	for (int i = 0; i < treeSize; i++)
	{
		insert(&root, new node(dist(gen)));
	}
	for (thrust::host_vector<int>::iterator it = h.begin(); it != h.end(); it++)
	{
		*it = dist(gen);
	}
#else
	for (int i = 0; i < treeSize; i++)
	{
		insert(&root, new node(rand() % (high -low) + low));
	}
	for (thrust::host_vector<int>::iterator it = h.begin(); it != h.end(); it++)
	{
		*it = rand() % (high -low) + low;
	}
#endif

	thrust::host_vector<bool> foundh(searchSize, false);

	thrust::device_vector<int> d(searchSize);
	thrust::device_vector<bool> foundd(searchSize, false);
	d = h;
	// TODO Die Zufallszahlen könnte man auch auf dem Device erzeugen.

	Extent ext(searchSize);
	CudaExecConfig cnf(ext, dim3(128, 1, 1));

	rec_find(cnf, root, d, foundd);
	cudaDeviceSynchronize();
	check_cuda();

	while_find(cnf, root, d, foundd);
	cudaDeviceSynchronize();
	check_cuda();
}
