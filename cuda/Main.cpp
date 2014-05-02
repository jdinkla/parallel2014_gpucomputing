/*
 * Copyright (c) 2014 by Joern Dinkla, www.dinkla.com, All rights reserved.
 *
 * See the LICENSE file in the root directory.
 */

#define _CRT_SECURE_NO_WARNINGS 1

#include <iostream>
#include "thrust_map_beispiel.h"
#include "FunctionCaller.h"
#include "MandelbrotBenchmark.h"
#include "recursive_kernel.h"
#include "while_find.h"
#include "TreeBenchmark.h"

using namespace std;

functions_t fs;

void synopsis()
{
	cout << "SYNOPSIS: cuda.exe ARG" << endl;
	cout << "  where ARG is one of { ";
	char* sep = "";
	for (functions_t::iterator it = fs.begin(); it != fs.end(); it++)
	{
		cout << sep << (*it).key;
		sep = ", ";
	}
	cout << " }" << endl;
}

int main(int argc, char** argv)
{
	fs.push_back(KeyFunctionPair("thrust_map", &thrust_map_beispiel));
	fs.push_back(KeyFunctionPair("mandelbrot", &bench_mandelbrot));
	fs.push_back(KeyFunctionPair("mandelbrot_short", &bench_mandelbrot_short));
	fs.push_back(KeyFunctionPair("mandelbrot_shorter", &bench_mandelbrot_shorter));
	fs.push_back(KeyFunctionPair("mandelbrot_128", &bench_mandelbrot_128));
	fs.push_back(KeyFunctionPair("mandelbrot_128_compare", &bench_mandelbrot_128_compare));
	fs.push_back(KeyFunctionPair("cuda_recursive_kernel", &cuda_recursive_kernel));
	fs.push_back(KeyFunctionPair("cuda_while_example_7", &cuda_while_example_7));
	fs.push_back(KeyFunctionPair("tree_compare_rec_while", &tree_compare_rec_while));

	FunctionCaller fc(fs, &synopsis);

	try
	{
		int rc = fc.exec(argc, argv);
		return rc;
	}
	catch (std::exception& e)
	{
		cerr << "ERROR: " << e.what() << endl;
	}
	return 0;
}