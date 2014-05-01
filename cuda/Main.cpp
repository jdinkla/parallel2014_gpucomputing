/*
 * Copyright (c) 2014 by Joern Dinkla, www.dinkla.com, All rights reserved.
 *
 * See the LICENSE file in the root directory.
 */

#include <iostream>
#include "Beispiele.h"

using namespace std;

void synopsis()
{
	cout << "ERROR: wrong number of arguments!" << endl;
	cout << "SYNOPSIS: cuda.exe ARG" << endl;
	cout << "  where ARG is one of { thrust_map } " << endl;
}

inline bool is_equal(char* ptr1, char* ptr2)
{
	return string(ptr1).compare(ptr2) == 0;
}

int main(int argc, char** argv)
{
	try
	{
		if (argc != 2)
		{
			synopsis();
			return 0;
		}
		else
		{
			if (is_equal(argv[1], "thrust_map"))
			{
				thrust_map_beispiel();
			}
			else
			{
				synopsis();
				return 0;
			}
		}
	}
	catch (std::exception& e)
	{
		cerr << "ERROR: " << e.what() << endl;
	}
	return 0;
}