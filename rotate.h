#pragma once
#include "cuda_img.h"

struct KernelRot {
	float s;
	float c;
};

void cu_run_rotate( const CudaImg &src, CudaImg &dst, const KernelRot &kr );
