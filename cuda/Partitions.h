/*
* Copyright (c) 2014 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#pragma once

#include <cuda_runtime_api.h>
#include "CudaUtilities.h"
#include "Partition.h"
#include <numeric>

class Partitions
{
public:

	Partitions(const int _numJobs, const int _numParts)
		: numJobs(_numJobs), numParts(_numParts)
	{
		partitions = new Partition[numParts];
		const int sizeJob = numJobs / numParts;
		int start = 0;
		for (int i = 0; i < numParts; i++)
		{
			Partition* p = &partitions[i];
			p->partition_id = i;
			p->device_id = i / 2;
			cudaSetDevice(p->device_id);										check_cuda();
			cudaStreamCreate(&p->stream);										check_cuda();
			p->start = start;
			p->end = std::min(sizeJob + sizeJob - 1, numJobs);
			start += sizeJob;
		}
	}

	~Partitions()
	{
		for (int i = 0; i < numParts; i++)
		{
			Partition* p = &partitions[i];
			cudaSetDevice(p->device_id);										check_cuda();
			cudaStreamDestroy(p->stream);										check_cuda();
		}

		delete[] partitions;
	}

	Partition* get(const int i)
	{
		return &partitions[i];
	}

	static Partition* create_partitions(const int numJobs, const int numParts)
	{
		Partition* partitions = new Partition[numParts];
		return partitions;
	}

private:
	Partition* partitions;
	const int numJobs;
	const int numParts;
};

