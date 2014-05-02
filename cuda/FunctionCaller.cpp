/*
* Copyright (c) 2014 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#include "FunctionCaller.h"
#include "Function.h"
#include <iostream>

using namespace std;

inline bool is_equal(char* ptr1, char* ptr2)
{
	return string(ptr1).compare(ptr2) == 0;
}

FunctionCaller::FunctionCaller(functions_t _fs, func_t _synopsis)
: fs(_fs), synopsis(_synopsis)
{
	fs.push_back(KeyFunctionPair("?", synopsis));
	fs.push_back(KeyFunctionPair("help", synopsis));
}

int FunctionCaller::exec(int argc, char** argv)
{
	if (argc != 2)
	{
		cout << "ERROR: wrong number of arguments!" << endl;
		synopsis();
		return 1;
	}
	else
	{
		bool called = false;
		for (functions_t::iterator it = fs.begin(); it != fs.end(); it++)
		{
			KeyFunctionPair f = *it;
			if (is_equal(argv[1], f.key))
			{
				called = true;
				(*f.func)();
			}
		}
		if (!called)
		{
			cout << "ERROR: unknown argument!" << endl;
			synopsis();
			return 1;
		}
	}
	return 0;
}