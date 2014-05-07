/*
* Copyright (c) 2014 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#pragma once

#include <vector>
#include <cuda_runtime_api.h>

typedef std::pair<int, int> pair2d;
typedef std::vector<pair2d> pairs2d;

pairs2d get_blocks();
pairs2d get_special_blocks();

size_t get_size_2d();
