#pragma once

#include <cuda_runtime_api.h>

// Überprüft den Zustand des Error-Flags von CUDA
void check_cuda();

// Gibt die Größe des freien Speichers in Bytes zurück.
size_t get_free_device_mem();

// Gibt die Größe des Speichers in Bytes zurück.
size_t get_total_device_mem();
