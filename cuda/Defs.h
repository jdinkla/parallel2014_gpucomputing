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
#undef WINDOWS
#undef WINDOWS_C11 
#undef WINDOWS_CPP
#undef LINUX
#undef LINUX_C11 
#undef LINUX_CPP
#undef MAC
#undef MAC_C11 
#undef MAC_CPP

// --------------- Windows ---------------
#ifdef _MSC_VER
#define WINDOWS 1
#if __cplusplus >= 201103L
#define WINDOWS_C11 1
#else
#define WINDOWS_CPP 1
#endif
#endif

// --------------- Mac ---------------
#ifdef __APPLE__
#define MAC 1
#if __cplusplus >= 201103L
#define MAC_C11 1
#else
#define MAC_CPP 1
#endif
#endif

// --------------- Linux ---------------
#ifdef __linux__
#define LINUX 1
#if __cplusplus >= 201103L
#define LINUX_C11 1
#else
#define LINUX_CPP 1
#endif
#endif

