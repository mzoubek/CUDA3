// #include <cuda_runtime_api.h>
#include <cuda_device_runtime_api.h>
// #include <cuda_runtime.h>
#include <stdlib.h>
#include <cmath>
#include "opencv2/opencv.hpp"

#include "uni_mem_allocator.h"
#include "cuda_img.h"
#include "rotate.h"

using namespace std;

extern void cuRotate( CudaImg cudaImg, CudaImg rotateImg, float sin, float cos );

int main( int argc, char **argv )
{
	if ( argc < 4 )
	{
		cerr << "Usage: " << argv[0] << " <input.png> <angle_degrees>" << endl;
	}

	UniformAllocator allocator;
	cv::Mat::setDefaultAllocator( &allocator );

	const char* inputPath = argv[ 1 ];
	float angleDeg = atof( argv[ 2 ] );
	float angleRad = angleDeg * static_cast<float>(CV_PI) / 180.0f;

	cv::Mat bgraImg = cv::imread( inputPath, cv::IMREAD_UNCHANGED );

	if ( bgraImg.empty() )
	{
		cerr << "Unable to load the image, please check the filepath" << endl;
		exit(-1);
	}
	cout << "Channels: " << bgraImg.channels() << endl;

	if ( bgraImg.type() != CV_8UC4 )
	{
		bgraImg.convertTo( bgraImg, CV_8UC4 );
	}

	cv::Mat bgraRotateImg( bgraImg.size().height, bgraImg.size().width, CV_8UC4 );

	CudaImg src;
	src.m_size.x = bgraImg.size().width;
	src.m_size.y = bgraImg.size().height;
	src.m_p_uchar4 = ( uchar4* ) bgraImg.data;

	CudaImg dst;
	dst.m_size.x = bgraRotateImg.size().width;
	dst.m_size.y = bgraRotateImg.size().height;
	dst.m_p_uchar4 = ( uchar4* ) bgraRotateImg.data;

	KernelRot kr{ sinf( angleRad ), cosf( angleRad ) };

	cu_run_rotate( src, dst, kr );

	cv::imshow( "Rotated Image", bgraRotateImg );
	cv::waitKey(0);

	return 0;
}
