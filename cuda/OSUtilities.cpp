/*
* Copyright (c) 2012-14 by Jörn Dinkla, www.dinkla.com, All rights reserved.
*/

#ifdef _MSC_VER
#include <windows.h>
#include <numeric>
#include <sstream>
#include <string>
#include <ctime>
#include <iomanip>
#endif

#include "OSUtilities.h"

using namespace std;

long long GetCurrentSystemTime()
{
#ifdef _MSC_VER
	// see http://en.allexperts.com/q/C-1040/time-milliseconds-Windows.htm
	FILETIME now;
	GetSystemTimeAsFileTime(&now);

	ULARGE_INTEGER uli;
	uli.LowPart = now.dwLowDateTime;
	uli.HighPart = now.dwHighDateTime;

	return (long long)(uli.QuadPart / 10000);
#elif __APPLE__
	return 1;
#else
	return 0;
#endif
}

