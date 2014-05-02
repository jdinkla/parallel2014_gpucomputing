/*
* Copyright (c) 2014 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#pragma once

#include "Function.h"

// TODO hier stimmt die Abstraktion mit der synopsis noch nicht richtig

class FunctionCaller
{
public:

	FunctionCaller(functions_t _fs, func_t _synopsis);

	int exec(int argc, char** argv);

private:
	functions_t fs;
	func_t synopsis;

};