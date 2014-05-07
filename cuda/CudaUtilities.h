#pragma once

#include <cuda_runtime_api.h>

// �berpr�ft den Zustand des Error-Flags von CUDA
void check_cuda();

// Gibt die Gr��e des freien Speichers in Bytes zur�ck.
size_t get_free_device_mem();

// Gibt die Gr��e des Speichers in Bytes zur�ck.
size_t get_total_device_mem();
