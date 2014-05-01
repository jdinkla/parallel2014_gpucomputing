#pragma once

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <iostream>

template <typename T>
void show(thrust::device_vector<T>& d)
{
	for (int i = 0; i < d.size(); i++)
	{
		std::cout << "d[" << i << "] = " << d[i] << std::endl;
	}
}

template <typename T>
void show(thrust::host_vector<T>& d)
{
	for (int i = 0; i < d.size(); i++)
	{
		std::cout << "d[" << i << "] = " << d[i] << std::endl;
	}
}
