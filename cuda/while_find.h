/*
* Copyright (c) 2014 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#pragma once

#include "node.h"
#include "CudaExecConfig.h"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

void while_find(
	CudaExecConfig cnf,
	const node* n,
	thrust::device_vector<int>& elems,
	thrust::device_vector<bool>& found
	);

void while_find_host(
	const node* n,
	thrust::host_vector<int>& elems,
	thrust::host_vector<bool>& found,
	const int x);
