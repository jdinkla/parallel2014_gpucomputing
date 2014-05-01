/*
 * Copyright (c) 2014 by Joern Dinkla, www.dinkla.com, All rights reserved.
 *
 * See the LICENSE file in the root directory.
 */

#include <iostream>
#include "Beispiele.h"

using namespace std;

int main(int argc, char** argv)
{
	try
	{
		amp_map_beispiel();
	}
	catch (std::exception& e)
	{
		cerr << "ERROR: " << e.what() << endl;
	}
	return 0;
}