/*
 * Copyright (c) 2014 by Joern Dinkla, www.dinkla.com, All rights reserved.
 *
 * See the LICENSE file in the root directory.
 */

#include <amp.h>
#include <iostream>
#include <algorithm>
#include <numeric>

using namespace std;
using namespace concurrency;

void amp_map_beispiel()
{
	std::vector<int> v(1024);
	std::iota(v.begin(), v.end(), 1);
	array_view<int, 1> av(1024, v);
	parallel_for_each(av.extent, 
	    [=](index<1> i) restrict(amp) 
	{
		av[i] = av[i] * 2;
	});
	av.synchronize();
	for (int i = 0; i < v.size(); i++) {
		cout << "v[" << i << "] = " << v[i] << endl;
	}

}









