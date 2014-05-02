/*
* Copyright (c) 2014 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#pragma once

struct Rectangle
{
	float x0, x1, y0, y1;

	Rectangle(float _x0, float _x1, float _y0, float _y1)
		: x0(_x0), x1(_x1), y0(_y0), y1(_y1)
	{
	}
};
