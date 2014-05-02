/*
* Copyright (c) 2014 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#pragma once

/*
	Weil NVIDIA CUDA 6.0 noch kein C++ 11 kann, insbesondere kein Lambda-Ausdrücke
	sind hier noch mal Function-Pointer zu finden.
*/

#include <vector>

typedef void(*func_t)();

struct KeyFunctionPair
{
	char* key;
	func_t func;
	KeyFunctionPair(char* _key = 0, func_t _func = 0) : key(_key), func(_func) {}
};

typedef std::vector<KeyFunctionPair> functions_t;

