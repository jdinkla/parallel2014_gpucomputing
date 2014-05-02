/*
* Copyright (c) 2014 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#pragma once

/* 
	Hier werden die Plattformen unterschieden
	Windows, Linux, Mac
	jeweils für C11 oder ohne C11
*/

#undef C11
#undef WINDOWS_C11 
#undef WINDOWS_CPP
#undef LINUX_C11 
#undef LINUX_CPP
#undef MAC_C11 
#undef MAC_CPP

#if __cplusplus == 201103L

#define C11 1

#ifdef _MSC_VER
#define WINDOWS_C11 1
#endif

#else 

#ifdef _MSC_VER
#define WINDOWS_CPP 1
#endif

#endif


