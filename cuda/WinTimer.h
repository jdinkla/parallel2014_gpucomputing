/*
* Copyright (c) 2014 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#pragma once

#include "ITimer.h"
#include "OSUtilities.h"

class Timer
	: public ITimer
{
public:

	Timer()
	{
	}

	void start()
	{
		start_val = GetCurrentSystemTime();
	}

	void stop()
	{
		stop_val = GetCurrentSystemTime();
	}

	float delta()
	{
		const long long dur = stop_val - start_val;
		return (float) dur;
	}

private:

	long long start_val;
	long long stop_val;

};
