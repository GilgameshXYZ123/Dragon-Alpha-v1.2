#pragma once

#ifndef IMG_TRANSPOSED_3D_H
#define IMG_TRANSPOSED_3D_H

//mul(Xdim) = length = mul(Ydim)
//lengthv = length / Xdim2 * stride
//stride % 4 == 0, so: lengthv % 4 == 0
#ifndef IMG_TRANSPOSED_3D_CALL
#define IMG_TRANSPOSED_3D_CALL

//LB = log2(BLOCK_SIZE)

//dimIdx2 < 2: lengthv % 16 == 0
#define img_tp3d_k16(stream, LB, LT, X, Y, Xd1, Yd1, dIdx1, dIdx2, stride, length)\
	img_transpose3D_kernel_16\
		<<< (lengthv >> LB >> LT), (1<<LB), 0, stream>>>\
			(X, Y, (Xd1*stride), Yd1, dIdx1, dIdx2, stride, length)

//dimIdx2 < 2: lengthv % 8 == 0
#define img_tp3d_k8(stream, LB, LT, X, Y, Xd1, Yd1, dIdx1, dIdx2, stride, length)\
	img_transpose3D_kernel_8\
		<<< (lengthv >> LB >> LT), (1<<LB), 0, stream>>>\
			(X, Y, (Xd1*stride), Yd1, dIdx1, dIdx2, stride, length)

//dimIdx2 < 2
#define img_tp3d_k4(stream, LB, LT, X, Y, Xd1, Yd1, dIdx1, dIdx2, stride, length)\
	img_transpose3D_kernel_4\
		<<< (lengthv >> LB >> LT), (1<<LB), 0, stream>>>\
			(X, Y, (Xd1*stride), Yd1, dIdx1, dIdx2, stride, length)

#define img_tp3d_k4_small(stream, X, Y, Xd1, Yd1, dIdx1, dIdx2, stride, lengthv)\
	img_transpose3D_kernel_4\
		<<< 1, (lengthv + 3) >> 2, 0, stream >>>\
			(X, Y, (Xd1*stride), Yd1, dIdx1, dIdx2, stride, lengthv)

//common
#define img_tp3d_k1(stream, LB, LT, X, Y, Xd1, Xd2, Yd1, Yd2, dIdx1, dIdx2, strideX, strideY, length)\
	img_transpose3D_kernel_1\
		<<< (length >> LB >> LT), (1<<LB), 0, stream>>>\
			(X, Y, (Xd1*Xd2), Xd2, Yd1, Yd2, dIdx1, dIdx2, strideX, strideY, length)

#define img_tp3d_k1_small(stream, X, Y, Xd1, Xd2, Yd1, Yd2, dIdx1, dIdx2, strideX, strideY, length)\
	img_transpose3D_kernel_1\
		<<< 1, (length + 3) >> 2, 0, stream >>>\
			(X, Y, (Xd1*Xd2), Xd2, Yd1, Yd2, dIdx1, dIdx2, strideX, strideY, length)

#endif


//======[dimIndex2 < 3]=================================
//lengthv % 16 == 0
#ifndef IMG_TRANSPOSE_3D_KERNEL_16
#define IMG_TRANSPOSE_3D_KERNEL_16

//if dimIndex2 < 2: 
//the tranpose is performed on the first two dim
//so the basic mem struture is not changed, and: Ydim2 = Xdim2, we can use char4

__global__ void img_transpose3D_kernel_16(
	const char* __restrict__ X,
	      char* __restrict__ Y,
	int Xdim12,//Xdim2 = strideX = stride(consider memory alignment)
	int Ydim1,//Ydim2 = strideY = stride
	int dimIndex1, int dimIndex2,
	int stride, int lengthv)
{
	int index = (blockIdx.x*blockDim.x) + threadIdx.x, index16 = (index << 4);
	int step = (blockDim.x * gridDim.x), step16 = step << 4;

	for (int x[3]; index16 < lengthv; index16 += step16)
	{
		int xoffset = index16;
		x[0] = xoffset / Xdim12; int xoffset_res = xoffset - x[0] * Xdim12;
		x[1] = xoffset_res / stride;//Xdim2 = stride
		x[2] = xoffset_res - x[1] * stride;//Xdim2 = stride

		int t = x[dimIndex1]; x[dimIndex1] = x[dimIndex2]; x[dimIndex2] = t;
		int yoffset = (x[0] * Ydim1 + x[1])*stride + x[2];//Xdim2 = stride

		*(char16*)(Y + yoffset) = *(char16*)(X + xoffset);
	}
}

#endif


//lengthv % 8 == 0
#ifndef IMG_TRANSPOSE_3D_KERNEL_8
#define IMG_TRANSPOSE_3D_KERNEL_8

//if dimIndex2 < 2: 
//the tranpose is performed on the first two dim
//so the basic mem struture is not changed, and: Ydim2 = Xdim2, we can use char4

__global__ void img_transpose3D_kernel_8(
	const char* __restrict__ X,
	      char* __restrict__ Y,
	int Xdim12,//Xdim2 = strideX = stride(consider memory alignment)
	int Ydim1,//Ydim2 = strideY = stride
	int dimIndex1, int dimIndex2,
	int stride, int lengthv)
{
	int index = (blockIdx.x*blockDim.x) + threadIdx.x, index8 = (index << 3);
	int step = (blockDim.x * gridDim.x), step8 = step << 3;

	for (int x[3]; index8 < lengthv; index8 += step8)
	{
		int xoffset = index8;
		x[0] = xoffset / Xdim12; int xoffset_res = xoffset - x[0] * Xdim12;
		x[1] = xoffset_res / stride;//Xdim2 = stride
		x[2] = xoffset_res - x[1] * stride;//Xdim2 = stride

		int t = x[dimIndex1]; x[dimIndex1] = x[dimIndex2]; x[dimIndex2] = t;
		int yoffset = (x[0] * Ydim1 + x[1])*stride + x[2];//Xdim2 = stride

		*(char8*)(Y + yoffset) = *(char8*)(X + xoffset);
	}
}

#endif


//lengthv % 4 == 0
#ifndef IMG_TRANSPOSE_3D_KERNEL_4
#define IMG_TRANSPOSE_3D_KERNEL_4

//if dimIndex2 < 2: 
//the tranpose is performed on the first two dim
//so the basic mem struture is not changed, and: Ydim2 = Xdim2£¬ we can use char4

__global__ void img_transpose3D_kernel_4(
	const char* __restrict__ X,
	      char* __restrict__ Y,
	int Xdim1,//Xdim2 = strideX = stride(consider memory alignment)
	int Ydim1,//Ydim2 = strideY = stride
	int dimIndex1, int dimIndex2,
	int stride, int lengthv)
{
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;
	int step = (blockDim.x * gridDim.x), step4 = step << 2;

	int Xdim12 = Xdim1 * stride;//Xdim2 = strideX = stride

	int x[3];
	for (int index4 = (index << 2); index4 < lengthv; index4 += step4)
	{
		int xoffset = index4;
		x[0] = xoffset / Xdim12; int xoffset_res = xoffset - x[0] * Xdim12;
		x[1] = xoffset_res / stride;//Xdim2 = stride
		x[2] = xoffset_res - x[1] * stride;//Xdim2 = stride

		int t = x[dimIndex1]; x[dimIndex1] = x[dimIndex2]; x[dimIndex2] = t;
		int yoffset = (x[0] * Ydim1 + x[1])*stride + x[2];//Xdim2 = stride

		*(char4*)(Y + yoffset) = *(char4*)(X + xoffset);
	}
}

#endif


//======[common]========================================
#ifndef IMG_TRANSPOSE_3D_KERNEL_1
#define IMG_TRANSPOSE_3D_KERNEL_1

//if dimIndex1 > dimIndex2: swap(dimIndex1, dimIndex2)
//so: dimIndex1 < dimIndex2
__global__ void img_transpose3D_kernel_1(
	const char* __restrict__ X,
	      char* __restrict__ Y,
	int Xdim12, int Xdim2,//Xdim0
	int Ydim1, int Ydim2,//Ydim0
	int dimIndex1, int dimIndex2,
	int strideX, int strideY, int length)
{
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;
	int step = (blockDim.x * gridDim.x);

	for (int x[3]; index < length; index += step)
	{
		int xoffset = index;
		x[0] = xoffset / Xdim12; int xoffset_res = xoffset - x[0] * Xdim12;
		x[1] = xoffset_res / Xdim2;
		x[2] = xoffset_res - x[1] * Xdim2;

		int t = x[dimIndex1]; x[dimIndex1] = x[dimIndex2]; x[dimIndex2] = t;
		int yoffset = (x[0] * Ydim1 + x[1])*Ydim2 + x[2];

		//consider the mem alignment
		const int xoft1 = xoffset / Xdim2, xoft0 = xoffset - xoft1 * Xdim2;
		const int yoft1 = yoffset / Ydim2, yoft0 = yoffset - yoft1 * Ydim2;
		Y[yoft1 * strideY + yoft0] = X[xoft1 * strideX + xoft0];
	}
}

#endif


void __img_transpose3d(cudaStream_t stream,
	const char* __restrict__ X,
	      char* __restrict__ Y,
	int Xd1, int Xd2,//Xdim0
	int Yd1, int Yd2,//Ydim0
	int dIdx1, int dIdx2,
	int strideX, int strideY, //the mem_stride >= mem_width
	int length)
{
	//make sure: dIdx2 > dIdx1
	if (dIdx1 > dIdx2) { int t = dIdx1; dIdx1 = dIdx2; dIdx2 = t; }

	//dimIndex2: we must have, aligned_Xdim2 = aligned_Ydim2 = strideX = strideY = stride
	if (dIdx2 < 2) {//no change to basic mem structure
		int lengthv = length / Xd2 * strideX;// length / mem_width * mem_stride
		if (lengthv < 256) { img_tp3d_k4_small(stream, X, Y, Xd1, Yd1, dIdx1, dIdx2, strideX, lengthv); return; }
		if (!(lengthv & 15) && lengthv >= 16384) { img_tp3d_k16(stream, 5, 4, X, Y, Xd1, Yd1, dIdx1, dIdx2, strideX, lengthv); return; }
		if (!(lengthv &  7) && lengthv >=  8192) { img_tp3d_k8(stream, 5, 3, X, Y, Xd1, Yd1, dIdx1, dIdx2, strideX, lengthv); return; }
		img_tp3d_k4(stream, 5, 2, X, Y, Xd1, Yd1, dIdx1, dIdx2, strideX, lengthv); return;
	}

	if ((dIdx2 == 2) && (dIdx1 == 1) && (Xd2 == strideX) && (Yd2 == strideY)) {
		int Batch = length / (Xd1 * Xd2);//Batch = Xd0
		int N = Xd1;
		int M = Xd2;
		if (__img_batch_mat_transpose(stream, X, Y, Batch, N, M)) return;
	}

	if (length < 256) { img_tp3d_k1_small(stream, X, Y, Xd1, Xd2, Yd1, Yd2, dIdx1, dIdx2, strideX, strideY, length); return; }
	img_tp3d_k1(stream, 5, 2, X, Y, Xd1, Xd2, Yd1, Yd2, dIdx1, dIdx2, strideX, strideY, length);
}

#endif