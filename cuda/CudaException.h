#pragma once

#include <exception>

// In C++ 11 hätte ich einfach throw std::runtime_error(str); verwendet.

class CudaException
	: public std::exception
{
public:

	CudaException(const char* _msg)
		: msg(_msg)
	{}

	const char* what() const
	{
		return msg;
	}

private:

	const char* msg;
	
};