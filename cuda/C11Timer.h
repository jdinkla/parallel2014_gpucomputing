/*
* Copyright (c) 2014 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#pragma once

#include <chrono>
#include <string>
#include <functional>
#include "ITimer.h"

class C11Timer
	: public ITimer
{
public:

	C11Timer()
	{
	}

	void start()
	{
		start_val = std::chrono::system_clock::now();
	}

	void stop()
	{
		stop_val = std::chrono::system_clock::now();
	}

	float delta()
	{
		return (float)getDuration().count();
	}

	std::chrono::milliseconds getDuration()
	{
		return std::chrono::duration_cast<std::chrono::milliseconds>(stop_val - start_val);
	}

private:

	std::chrono::system_clock::time_point start_val;
	std::chrono::system_clock::time_point stop_val;

};

