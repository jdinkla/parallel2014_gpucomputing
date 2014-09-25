/*
 * Copyright (c) 2014 by Joern Dinkla, www.dinkla.com, All rights reserved.
 *
 * See the LICENSE file in the root directory.
 */

#include <algorithm>				// CUDA 6.5 requires this for std::min
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>
#include <iostream>

using namespace std;

struct double_functor {
	__device__ int operator()(const int value) {
		return value * 2;
}};

void thrust_map_beispiel() {
	double_functor f;
	thrust::device_vector<int> d(1024);
	thrust::sequence(d.begin(), d.end(), 1);
	thrust::transform(d.begin(), d.end(), d.begin(), f);
	for (int i = 0; i < d.size(); i++) {
		cout << "d[" << i << "] = " << d[i] << endl;
}}





