/*
* Copyright (c) 2014 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#include "TreeBenchmark.h"
#include <random>
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

	// Erstelle Baum
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<> dist(low, high);
	node* root = 0;
	for (int i = 0; i < treeSize; i++)
	{
		insert(&root, new node(dist(gen)));
	}

	// Erstelle Such-Array
	thrust::host_vector<int> h(searchSize);
	for (thrust::host_vector<int>::iterator it = h.begin(); it != h.end(); it++)
	{
		*it = dist(gen);
	}
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
