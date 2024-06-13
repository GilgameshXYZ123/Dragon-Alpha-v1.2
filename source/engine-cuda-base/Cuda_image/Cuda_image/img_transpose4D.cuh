#pragma once

#ifndef IMG_TRANSPOSED_4D_H
#define IMG_TRANSPOSED_4D_H

//mul(Xdim) = length = mul(Ydim)
//lengthv = length / Xdim3 * stride
//stride % 4 == 0, so: lengthv % 4 == 0
#ifndef IMG_TRANSPOSED_4D_CALL
#define IMG_TRANSPOSED_4D_CALL

//LB = log2(BLOCK_SIZE)

//dimIdx2 < 3: lengthv % 16 == 0
#define img_tp4d_k16(stream, LB, LT, X, Y, Xd1, Xd2, Yd1, Yd2, dIdx1, dIdx2, stride, lengthv)\
	img_transpose4D_kernel_16\
		<<< (lengthv >> LB >> LT), (1<<LB), 0, stream>>>\
			(X, Y, (Xd1*Xd2*stride), (Xd2*stride), Yd1, Yd2, dIdx1, dIdx2, stride, lengthv)

//dimIdx2 < 3: lengthv % 8 == 0
#define img_tp4d_k8(stream, LB, LT, X, Y, Xd1, Xd2, Yd1, Yd2, dIdx1, dIdx2, stride, lengthv)\
	img_transpose4D_kernel_8\
		<<< (lengthv >> LB >> LT), (1<<LB), 0, stream>>>\
			(X, Y, (Xd1*Xd2*stride), (Xd2*stride), Yd1, Yd2, dIdx1, dIdx2, stride, lengthv)

//dimIdx2 < 3
#define img_tp4d_k4(stream, LB, LT, X, Y, Xd1, Xd2, Yd1, Yd2, dIdx1, dIdx2, stride, lengthv)\
	img_transpose4D_kernel_4\
		<<< (lengthv >> LB >> LT), (1<<LB), 0, stream>>>\
			(X, Y, (Xd1*Xd2*stride), (Xd2*stride), Yd1, Yd2, dIdx1, dIdx2, stride, lengthv)

#define img_tp4d_k4_small(stream, X, Y, Xd1, Xd2, Yd1, Yd2, dIdx1, dIdx2, stride, lengthv)\
	img_transpose4D_kernel_4\
		<<< 1, (lengthv + 3) >> 2, 0, stream >>>\
			(X, Y, (Xd1*Xd2*stride), (Xd2*stride), Yd1, Yd2, dIdx1, dIdx2, stride, lengthv)

//common
#define img_tp4d_k1(stream, LB, LT, X, Y, Xd1, Xd2, Xd3, Yd1, Yd2, Yd3, dIdx1, dIdx2, strideX, strideY, length)\
	img_transpose4D_kernel_1\
		<<< (length >> LB >> LT), (1<<LB), 0, stream>>>\
			(X, Y, (Xd1*Xd2*Xd3), (Xd2*Xd3), Xd3, Yd1, Yd2, Yd3, dIdx1, dIdx2, strideX, strideY, length)

#define img_tp4d_k1_small(stream, X, Y, Xd1, Xd2, Xd3, Yd1, Yd2, Yd3, dIdx1, dIdx2, strideX, strideY, length)\
	img_transpose4D_kernel_1\
		<<< 1, (length + 3) >> 2, 0, stream >>>\
			(X, Y, (Xd1*Xd2*Xd3), (Xd2*Xd3), Xd3, Yd1, Yd2, Yd3, dIdx1, dIdx2, strideX, strideY, length)

#endif


//======[dimIndex2 < 3]=================================
//lengthv % 16 == 0
#ifndef IMG_TRANSPOSE_4D_KERNEL_16
#define IMG_TRANSPOSE_4D_KERNEL_16

//if dimIndex2 < 3: 
//the tranpose is performed on the first three dim
//so the basic mem struture is not changed, and: Ydim3 = Xdim3, we can use char4
//(5, 4): Size = 32, Time = 0.088 mesc, Speed = 355.114 GB/s
__global__ void img_transpose4D_kernel_16(
	const char* __restrict__ X,
	char* __restrict__ Y,
	int Xdim123, int Xdim23,//Xdim3 = strideX = stride, (consider memory alignment)
	int Ydim1, int Ydim2,//Ydim3 = strideY = stride
	int dimIndex1, int dimIndex2,
	int stride, int lengthv)
{
	int index = (blockIdx.x*blockDim.x) + threadIdx.x, index16 = (index << 4);
	int step = (blockDim.x * gridDim.x), step16 = step << 4;

	for (int x[4]; index16 < lengthv; index16 += step16)
	{
		int xoffset = index16;
		x[0] = xoffset / Xdim123; int xoffset_res = xoffset - x[0] * Xdim123;
		x[1] = xoffset_res / Xdim23; xoffset_res -= x[1] * Xdim23;
		x[2] = xoffset_res / stride;//Xdim3 = stride
		x[3] = xoffset_res - x[2] * stride;//Xdim3 = stride

		int t = x[dimIndex1]; x[dimIndex1] = x[dimIndex2]; x[dimIndex2] = t;
		int yoffset = ((x[0] * Ydim1 + x[1])*Ydim2 + x[2])*stride + x[3];

		*(char16*)(Y + yoffset) = *(char16*)(X + xoffset);
	}
}

#endif


//lengthv % 8 == 0
#ifndef IMG_TRANSPOSE_4D_KERNEL_8
#define IMG_TRANSPOSE_4D_KERNEL_8

//if dimIndex2 < 3: 
//the tranpose is performed on the first three dim
//so the basic mem struture is not changed, and: Ydim3 = Xdim3, we can use char4
//(5, 3): Size = 32, Time = 0.098 mesc, Speed = 318.878 GB/s
__global__ void img_transpose4D_kernel_8(
	const char* __restrict__ X,
	      char* __restrict__ Y,
	int Xdim123, int Xdim23,//Xdim3 = strideX = stride, (consider memory alignment)
	int Ydim1, int Ydim2,//Ydim3 = strideY = stride
	int dimIndex1, int dimIndex2,
	int stride, int lengthv)
{
	int index = (blockIdx.x*blockDim.x) + threadIdx.x, index8 = (index << 3);
	int step = (blockDim.x * gridDim.x), step8 = step << 3;

	for (int x[4]; index8 < lengthv; index8 += step8)
	{
		int xoffset = index8;
		x[0] = xoffset / Xdim123; int xoffset_res = xoffset - x[0] * Xdim123;
		x[1] = xoffset_res / Xdim23; xoffset_res -= x[1] * Xdim23;
		x[2] = xoffset_res / stride;//Xdim3 = stride
		x[3] = xoffset_res - x[2] * stride;//Xdim3 = stride

		int t = x[dimIndex1]; x[dimIndex1] = x[dimIndex2]; x[dimIndex2] = t;
		int yoffset = ((x[0] * Ydim1 + x[1])*Ydim2 + x[2])*stride + x[3];

		*(char8*)(Y + yoffset) = *(char8*)(X + xoffset);
	}
}

#endif


//lengthv % 4 == 0
#ifndef IMG_TRANSPOSE_4D_KERNEL_4
#define IMG_TRANSPOSE_4D_KERNEL_4

//if dimIndex2 < 3: 
//the tranpose is performed on the first three dim
//so the basic mem struture is not changed, and: Ydim3 = Xdim3, we can use char4
//(5, 2): Size = 32, Time = 0.169 mesc, Speed = 184.911 GB/s
//(5, 3): Size = 32, Time = 0.135 mesc, Speed = 231.481 GB/s
__global__ void img_transpose4D_kernel_4(
	const char* __restrict__ X,
	      char* __restrict__ Y,
	int Xdim123, int Xdim23,//Xdim3 = strideX = stride, (consider memory alignment)
	int Ydim1, int Ydim2,//Ydim3 = strideY = stride
	int dimIndex1, int dimIndex2,
	int stride, int lengthv)
{
	int index = (blockIdx.x*blockDim.x) + threadIdx.x, index4 = (index << 2);
	int step = (blockDim.x * gridDim.x), step4 = step << 2;

	for (int x[4]; index4 < lengthv; index4 += step4)
	{
		int xoffset = index4;
		x[0] = xoffset / Xdim123; int xoffset_res = xoffset - x[0] * Xdim123;
		x[1] = xoffset_res / Xdim23; xoffset_res -= x[1] * Xdim23;
		x[2] = xoffset_res / stride;//Xdim3 = stride
		x[3] = xoffset_res - x[2] * stride;//Xdim3 = stride

		int t = x[dimIndex1]; x[dimIndex1] = x[dimIndex2]; x[dimIndex2] = t;
		int yoffset = ((x[0] * Ydim1 + x[1])*Ydim2 + x[2])*stride + x[3];

		*(char4*)(Y + yoffset) = *(char4*)(X + xoffset);
	}
}

#endif


//======[common]========================================
#ifndef IMG_TRANSPOSE_4D_KERNEL_1
#define IMG_TRANSPOSE_4D_KERNEL_1

//if dimIndex1 > dimIndex2: swap(dimIndex1, dimIndex2)
//so: dimIndex1 < dimIndex2
//(5, 2): Size = 32, Time = 0.368 mesc, Speed = 84.9185 GB/s
//(5, 3): Size = 32, Time = 0.329 mesc, Speed = 94.9848 GB/s
__global__ void img_transpose4D_kernel_1(
	const char* __restrict__ X,
	      char* __restrict__ Y,
	int Xdim123, int Xdim23, int Xdim3, //Xdim0
	int Ydim1, int Ydim2, int Ydim3, //Ydim0
	int dimIdx1, int dimIdx2,
	int strideX, int strideY, int length)
{
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;
	int step = (blockDim.x * gridDim.x);

	for (int x[4]; index < length; index += step)
	{
		int xoffset = index;
		x[0] = xoffset / Xdim123; int xoffset_res = xoffset - x[0] * Xdim123;
		x[1] = xoffset_res / Xdim23; xoffset_res -= x[1] * Xdim23;
		x[2] = xoffset_res / Xdim3;
		x[3] = xoffset_res - x[2] * Xdim3;

		int t = x[dimIdx1]; x[dimIdx1] = x[dimIdx2]; x[dimIdx2] = t;
		int yoffset = ((x[0] * Ydim1 + x[1])*Ydim2 + x[2])*Ydim3 + x[3];

		//consider the mem alignment
		const int xoft1 = xoffset / Xdim3, xoft0 = xoffset - xoft1 * Xdim3;
		const int yoft1 = yoffset / Ydim3, yoft0 = yoffset - yoft1 * Ydim3;
		Y[yoft1 * strideY + yoft0] = X[xoft1 * strideX + xoft0];
	}
}

#endif


void __img_transpose4d(cudaStream_t stream,
	const char* __restrict__ X,
	      char* __restrict__ Y,
	int Xd1, int Xd2, int Xd3,//Xdim0
	int Yd1, int Yd2, int Yd3,//Ydim0
	int dIdx1, int dIdx2,
	int strideX, int strideY,//the mem_stride >= mem_width
	int length)
{
	//make sure: dIdx2 > dIdx1
	if (dIdx1 > dIdx2) { int t = dIdx1; dIdx1 = dIdx2; dIdx2 = t; }

	if (dIdx2 < 3) {//dimIndex2: we must have, Xdim3 = Ydim3
		int lengthv = (length / Xd3 * strideX);// length / mem_width * mem_stride
		if (lengthv < 256) { img_tp4d_k4_small(stream, X, Y, Xd1, Xd2, Yd1, Yd2, dIdx1, dIdx2, strideX, lengthv); return; }
		if (!(lengthv & 15) && lengthv >= 16384) { img_tp4d_k16(stream, 5, 4, X, Y, Xd1, Xd2, Yd1, Yd2, dIdx1, dIdx2, strideX, lengthv); return; }
		if (!(lengthv &  7) && lengthv >=  8192) { img_tp4d_k8(stream, 5, 3, X, Y, Xd1, Xd2, Yd1, Yd2, dIdx1, dIdx2, strideX, lengthv); return; }
		if (lengthv >= 8192) { img_tp4d_k4(stream, 5, 3, X, Y, Xd1, Xd2, Yd1, Yd2, dIdx1, dIdx2, strideX, lengthv); return; }
		img_tp4d_k4(stream, 5, 2, X, Y, Xd1, Xd2, Yd1, Yd2, dIdx1, dIdx2, strideX, lengthv); return;
	}

	if ((dIdx2 == 3) && (dIdx1 == 2) && (Xd3 == strideX) && (Yd3 == strideY)) {
		int Batch = length / (Xd2 * Xd3);//Batch = Xd0 * Xd1
		int N = Xd2;
		int M = Xd3;
		if (__img_batch_mat_transpose(stream, X, Y, Batch, N, M)) return;
	}

	if (length < 256) { img_tp4d_k1_small(stream, X, Y, Xd1, Xd2, Xd3, Yd1, Yd2, Yd3, dIdx1, dIdx2, strideX, strideY, length); return; }
	if (length >= 8192) { img_tp4d_k1(stream, 5, 3, X, Y, Xd1, Xd2, Xd3, Yd1, Yd2, Yd3, dIdx1, dIdx2, strideX, strideY, length); return; }
	img_tp4d_k1(stream, 5, 2, X, Y, Xd1, Xd2, Xd3, Yd1, Yd2, Yd3, dIdx1, dIdx2, strideX, strideY, length);
}

#endif