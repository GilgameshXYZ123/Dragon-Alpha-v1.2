#pragma once

#ifndef TENSOR_TO_PIX_2D_H
#define TENSOR_TO_PIX_2D_H

//lengthv = height * stride
//stride = (width + 3)/4*4
//[lenghtv, stride] % 4 == 0
//Time = 0.088 mesc, Speed = 88.7784GB / s
#ifndef TENSOR_TO_PIX_2D_CALL
#define TENSOR_TO_PIX_2D_CALL

//LB = log2(BLOCK_SIZE)
//LT = log2(THREAD_ELEMENT_VISIT_SIZE)

//lengthv < 256
#define tensor2pix2d_k4_small(stream, X, Y, lengthv, width, stride)\
	tensor2pix2D_kernel_4\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(X, Y, lengthv, width, stride)

//common
#define tensor2pix2d_k4(stream, LB, LT, X, Y, lengthv, width, stride)\
	tensor2pix2D_kernel_4\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(X, Y, lengthv, width, stride)

//lengthv > lengthv_max
#define tensor2pix2d_k4_max(stream, X, Y, lengthv, width, stride)\
	tensor2pix2D_kernel_4\
		<<< GRID_MAX, 32, 0, stream >>>\
			(X, Y, lengthv, width, stride)

#endif


#ifndef TENSOR_TO_PIX_2D_KERNEL
#define TENSOR_TO_PIX_2D_KERNEL

//x belongs-to: (-inf, 0] -> 0
//x belongs-to: (0, 1) -> 255*x
//x belongs-to: [1, inf) -> 255

__global__ void tensor2pix2D_kernel_4(
	const float* __restrict__ X,
	char* __restrict__ Y,
	int lengthv, int width, int stride)
{
	const int step = gridDim.x*blockDim.x, step4 = step << 2;
	const int index = blockIdx.x*blockDim.x + threadIdx.x;

	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		float4 x = *(float4*)(X + index4);

		uchar4 y;
		y.x = float_to_pixel(x.x);
		y.y = float_to_pixel(x.y);
		y.z = float_to_pixel(x.z);
		y.w = float_to_pixel(x.w);

		within_width(y, index4, stride, width);
		*(uchar4*)(Y + index4) = y;
	}
}

#endif


void __tensor2pix2D(cudaStream_t stream,
	const float* X,
	char* Y,
	int lengthv, int width, int stride)
{
	if (lengthv < 256) { tensor2pix2d_k4_small(stream, X, Y, lengthv, width, stride); return; }
	if (lengthv > LENGTHV_MAX) { tensor2pix2d_k4_max(stream, X, Y, lengthv, width, stride); return; }
	tensor2pix2d_k4(stream, 5, 2, X, Y, lengthv, width, stride);
}

#endif