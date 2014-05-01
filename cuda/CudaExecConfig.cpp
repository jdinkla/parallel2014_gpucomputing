#include "CudaExecConfig.h"

// Ganzzahlige Division, die immer nach oben rundet
// ceiling(x/y), z. B. ceiling_div(5, 2) = (5 + 1) / 2 = 3
int ceiling_div(int x, int y)
{
	return (x + y - 1) / y;
}

CudaExecConfig::CudaExecConfig(const Extent& extent, const dim3 _block, const int _shared_mem, cudaStream_t _stream)
: block(_block)
, grid(ceiling_div(extent.get_width(), _block.x),
	   ceiling_div(extent.get_height(), _block.y),
	   1)
, shared_mem(_shared_mem)
, stream(_stream)
{
}
