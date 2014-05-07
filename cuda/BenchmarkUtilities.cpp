/*
* Copyright (c) 2014 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#include "BenchmarkUtilities.h"
#include <iostream>
#include "CudaUtilities.h"
#include <cmath>

using namespace std;

pairs2d get_blocks()
{
	pairs2d result;
	result.push_back(pair2d(32, 1));
	result.push_back(pair2d(32, 2));
	result.push_back(pair2d(32, 3));
	result.push_back(pair2d(32, 4));
	result.push_back(pair2d(32, 5));
	result.push_back(pair2d(32, 6));
	result.push_back(pair2d(64, 1));
	result.push_back(pair2d(64, 2));
	result.push_back(pair2d(64, 3));
	result.push_back(pair2d(64, 4));
	result.push_back(pair2d(96, 1));
	result.push_back(pair2d(96, 2));
	result.push_back(pair2d(96, 3));
	result.push_back(pair2d(128, 1));
	result.push_back(pair2d(128, 2));
	result.push_back(pair2d(128, 3));
	result.push_back(pair2d(128, 4));
	result.push_back(pair2d(192, 1));
	result.push_back(pair2d(192, 2));
	result.push_back(pair2d(192, 3));
	result.push_back(pair2d(256, 1));
	result.push_back(pair2d(256, 2));
	result.push_back(pair2d(256, 3));
	result.push_back(pair2d(256, 4));
	result.push_back(pair2d(512, 1));
	result.push_back(pair2d(512, 2));
	result.push_back(pair2d(1024, 1));
	return result;
}

pairs2d get_special_blocks()
{
	pairs2d result;
	result.push_back(pair2d(128, 1));
	result.push_back(pair2d(256, 1));
	result.push_back(pair2d(192, 1));
	result.push_back(pair2d(512, 1));
	result.push_back(pair2d(96, 1));
	result.push_back(pair2d(1024, 1));
	result.push_back(pair2d(64, 1));
	return result;
}

size_t get_size_2d() 
{
	const size_t free = get_free_device_mem();
	cout << "free memory " << free << endl;
	const size_t free2 = free / 2; // Sicherheit
	const size_t free3 = free2 / 2; // zwei buffer
	const size_t free4 = free3 / 4; // int braucht 4 bytes
	const size_t squared = sqrt(free4);
	cout << "Using sizeX=sizeY=" << squared << endl;
	return squared;
}
