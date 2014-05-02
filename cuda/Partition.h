/*
* Copyright (c) 2014 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#pragma once

#include <cuda_runtime_api.h>

struct Partition
{
	int partition_id;
	int device_id;
	cudaStream_t stream;
	int start;
	int end;
};