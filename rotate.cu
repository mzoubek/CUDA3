#include "cuda_img.h"
#include "rotate.h"
#include <cuda_runtime.h>

__global__
void kernel_rotate( const CudaImg src, CudaImg dst, KernelRot kr )
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if ( x >= dst.m_size.x || y >= dst.m_size.y ) return;

	// Souradnice vuci stredu
	int cx = x - dst.m_size.x / 2;
	int cy = y - dst.m_size.y / 2;

	// Rotace
	float ox = kr.c * cx - kr.s * cy;
	float oy = kr.s * cx + kr.c * cy;

	// zpet do src-coords
	int sx = int( ox + src.m_size.x / 2 );
	int sy = int( oy + src.m_size.y / 2 );
	if ( sx < 0 || sx >= src.m_size.x || sy < 0 || sy >= src.m_size.y ) return;

	int pixelDst = y * dst.m_size.x + x;
	int pixelSrc = sy * src.m_size.x + x;
	dst.m_p_uchar4[ pixelDst ] = src.m_p_uchar4[ pixelSrc ];
}

void cu_run_rotate( const CudaImg &src, CudaImg &dst, const KernelRot &kr )
{
	cudaError_t cerr;

	dim3 threads( 32, 32 );
	dim3 blocks( ( dst.m_size.x + threads.x - 1 ) / threads.x,
			( dst.m_size.y + threads.y - 1 ) / threads.y );
	kernel_rotate<<<blocks, threads>>>( src, dst, kr );
	
	if ( ( cerr = cudaGetLastError() ) != cudaSuccess )
	{
		printf( "CUDA error[%d]: %s\n", __LINE__, cudaGetErrorString( cerr ) );
	}

	cudaDeviceSynchronize();
}
