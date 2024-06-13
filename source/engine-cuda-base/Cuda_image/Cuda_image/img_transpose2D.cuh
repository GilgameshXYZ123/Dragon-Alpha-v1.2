#pragma once

#ifndef IMG_TRANSPOSED_2D_H
#define IMG_TRANSPOSED_2D_H

//mul(Xdim) = length = mul(Ydim)
//Obviously: dIdx1 = 0, dIdx2 = 1
//stride % 4 == 0, so: lengthv % 4 == 0
#ifndef IMG_TRANSPOSED_2D_CALL
#define IMG_TRANSPOSED_2D_CALL

//LB = log2(BLOCK_SIZE)

#define img_tp2d_k1(stream, LB, LT, X, Y, Xd1, Yd1, strideX, strideY, length)\
	img_transpose2D_kernel_1\
		<<< (length >> LB >> LT), (1<<LB), 0, stream>>>\
			(X, Y, Xd1, Yd1, strideX, strideY, length)

#define img_tp2d_k1_small(stream, X, Y, Xd1, Yd1, strideX, strideY, length)\
	img_transpose2D_kernel_1\
		<<< 1, (length + 3) >> 2, 0, stream >>>\
			(X, Y, Xd1, Yd1, strideX, strideY, length)

#endif


//common
#ifndef IMG_TRANSPOSE_2D_KERNEL_1
#define IMG_TRANSPOSE_2D_KERNEL_1

//Y = X^T
//if dimIndex1 > dimIndex2: swap(dimIndex1, dimIndex2)
//so: dimIndex1 < dimIndex2
__global__ void img_transpose2D_kernel_1(
	const char* __restrict__ X,
	      char* __restrict__ Y,
	int Xdim1,//Ydim1 = Xdim0
	int Ydim1,//Ydim0 = Xdim1
	int strideX, int strideY, int length)
{
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;
	int step = (blockDim.x * gridDim.x);

	for (; index < length; index += step)
	{
		int xoffset = index;
		int x0 = xoffset / Xdim1;
		int x1 = xoffset - x0 * Xdim1;
		
		Y[x1 * strideY + x0] = X[x0 * strideX + x1];//swap(x1, x0)
	}
}

#endif


void __img_transpose2d(cudaStream_t stream,
	const char* __restrict__ X,
		  char* __restrict__ Y,
	int Xd1,//Xdim0 = Ydim1
	int Yd1,//Ydim0 = Xdim1
	int strideX, int strideY, //the mem_stride >= mem_width
	int length)
{
	int N = strideY;//include the padded 0, so Xdim0 = Ydim1 -> strideY
	int M = strideX;//include the padded 0, so Xdim1 -> strideX
	if (__img_mat_transpose(stream, X, Y, N, M)) return;

	if (length < 256) { img_tp2d_k1_small(stream, X, Y, Xd1, Yd1, strideX, strideY, length); return; }
	if (length >= 8192) { img_tp2d_k1(stream, 5, 3, X, Y, Xd1, Yd1, strideX, strideY, length); return; }
	img_tp2d_k1(stream, 5, 2, X, Y, Xd1, Yd1, strideX, strideY, length);
}

#endif
