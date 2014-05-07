#pragma once

#include <exception>
#include "Defs.h"

// In C++ 11 hätte ich einfach throw std::runtime_error(str); verwendet.

class CudaException
	: public std::exception
{
public:

	CudaException(const char* _msg)
		: msg(_msg)
	{}

#ifdef MAC
	virtual const char* what() const throw()
	{
		return msg;
	}
#else
	const char* what() const
	{
		return msg;
	}
#endif

private:

	const char* msg;
	
};