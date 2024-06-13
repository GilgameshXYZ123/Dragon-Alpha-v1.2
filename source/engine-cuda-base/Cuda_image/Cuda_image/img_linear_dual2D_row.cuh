#pragma once

#ifndef IMG_LINEAR_DUAL_2D_ROW_H
#define IMG_LINEAR_DUAL_2D_ROW_H

//lengthv = height * stride
//stride = (width + 3)/4*4
//[lengthv, row_lengthv] % stride == 0
//field_length * row_lengthv = X1.lengthv
#ifndef IMG_LINEAR_DUAL_2D_ROW_CALL
#define IMG_LINEAR_DUAL_2D_ROW_CALL

//LB = log2(BLOCK_SIZE)
//LT = log2(THREAD_ELEMENT_VISIT_SIZE)

//lengthv % 16 == 0
#define img_linear_dual2d_row_k16(stream, LB, LT, X1, X2, row_lengthv, alpha, beta, gamma, Y, lengthv, width, stride)\
	img_linear_dual2D_row_kernel_16\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(X1, X2, row_lengthv, alpha, beta, gamma, Y, lengthv, width, stride)

//lengthv % 8 == 0
#define img_linear_dual2d_row_k8(stream, LB, LT, X1, X2, row_lengthv, alpha, beta, gamma, Y, lengthv, width, stride)\
	img_linear_dual2D_row_kernel_8\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(X1, X2, row_lengthv, alpha, beta, gamma, Y, lengthv, width, stride)

//common
#define img_linear_dual2d_row_k4(stream, LB, LT, X1, X2, row_lengthv, alpha, beta, gamma, Y, lengthv, width, stride)\
	img_linear_dual2D_row_kernel_4\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(X1, X2, row_lengthv, alpha, beta, gamma, Y, lengthv, width, stride)

//common
#define img_linear_dual2d_row_k4_small(stream, X1, X2, row_lengthv, alpha, beta, gamma, Y, lengthv, width, stride)\
	img_linear_dual2D_row_kernel_4\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(X1, X2, row_lengthv, alpha, beta, gamma, Y, lengthv, width, stride)

#endif


//lengthv % 16 == 0
#ifndef IMG_LINEAR_DUAL_2D_ROW_KERNEL16
#define IMG_LINEAR_DUAL_2D_ROW_KERNEL16

//for each row[i]: Y[i] = alpha*X1[i] + beta*X2 + gamma
//(5, 3): Size = 16.0078, Time = 0.055 mesc, Speed = 284.23 GB/s
__global__ void img_linear_dual2D_row_kernel_16(
	const char*  __restrict__ X1,
	const float* __restrict__ X2, int row_lengthv,
	float alpha, float beta, float gamma,
	char* __restrict__ Y,
	int lengthv, int width, int stride)
{
	int step = gridDim.x*blockDim.x, step16 = step << 4;
	int index = blockIdx.x*blockDim.x + threadIdx.x;

	for (int index16 = index << 4; index16 < lengthv; index16 += step16)
	{
		uchar16 x1 = *(uchar16*)(X1 + index16);
		float4 x2;//X2_lengthv % 4 == 0

		float16 fy;//y = alpha*x1 + beta*x2 + gamma
		x2 = *(float4*)(X2 + (index16) % row_lengthv);
		fy.x0 = alpha * x1.x0 + beta * x2.x + gamma;
		fy.y0 = alpha * x1.y0 + beta * x2.y + gamma;
		fy.z0 = alpha * x1.z0 + beta * x2.z + gamma;
		fy.w0 = alpha * x1.w0 + beta * x2.w + gamma;

		x2 = *(float4*)(X2 + (index16 + 4) % row_lengthv);
		fy.x1 = alpha * x1.x1 + beta * x2.x + gamma;
		fy.y1 = alpha * x1.y1 + beta * x2.y + gamma;
		fy.z1 = alpha * x1.z1 + beta * x2.z + gamma;
		fy.w1 = alpha * x1.w1 + beta * x2.w + gamma;

		x2 = *(float4*)(X2 + (index16 + 8) % row_lengthv);
		fy.x2 = alpha * x1.x2 + beta * x2.x + gamma;
		fy.y2 = alpha * x1.y2 + beta * x2.y + gamma;
		fy.z2 = alpha * x1.z2 + beta * x2.z + gamma;
		fy.w2 = alpha * x1.w2 + beta * x2.w + gamma;

		x2 = *(float4*)(X2 + (index16 + 12) % row_lengthv);
		fy.x3 = alpha * x1.x3 + beta * x2.x + gamma;
		fy.y3 = alpha * x1.y3 + beta * x2.y + gamma;
		fy.z3 = alpha * x1.z3 + beta * x2.z + gamma;
		fy.w3 = alpha * x1.w3 + beta * x2.w + gamma;
		uchar16 y; PIXEL_CLIP_16(y, fy);

		within_width16(y, index16, stride, width);
		*(uchar16*)(Y + index16) = y;
	}
}

#endif


//lengthv % 8 == 0
#ifndef IMG_LINEAR_DUAL_2D_ROW_KERNEL8
#define IMG_LINEAR_DUAL_2D_ROW_KERNEL8

//for each row[i]: Y[i] = alpha*X1[i] + beta*X2 + gamma
//(5, 3): Size = 16.0078, Time = 0.064 mesc, Speed = 244.26 GB/s
__global__ void img_linear_dual2D_row_kernel_8(
	const char*  __restrict__ X1,
	const float* __restrict__ X2, int row_lengthv,
	float alpha, float beta, float gamma,
	char* __restrict__ Y,
	int lengthv, int width, int stride)
{
	int step = gridDim.x*blockDim.x, step8 = step << 3;
	int index = blockIdx.x*blockDim.x + threadIdx.x;

	for (int index8 = index << 3; index8 < lengthv; index8 += step8)
	{
		uchar8 x1 = *(uchar8*)(X1 + index8);
		float4 x2;//X2_lengthv % 4 == 0

		float8 fy; //y = alpha*x1 + beta*x2 + gamma
		x2 = *(float4*)(X2 + (index8) % row_lengthv);
		fy.x0 = alpha * x1.x0 + beta * x2.x + gamma;
		fy.y0 = alpha * x1.y0 + beta * x2.y + gamma;
		fy.z0 = alpha * x1.z0 + beta * x2.z + gamma;
		fy.w0 = alpha * x1.w0 + beta * x2.w + gamma;

		x2 = *(float4*)(X2 + (index8 + 4) % row_lengthv);
		fy.x1 = alpha * x1.x1 + beta * x2.x + gamma;
		fy.y1 = alpha * x1.y1 + beta * x2.y + gamma;
		fy.z1 = alpha * x1.z1 + beta * x2.z + gamma;
		fy.w1 = alpha * x1.w1 + beta * x2.w + gamma;
		uchar8 y; PIXEL_CLIP_8(y, fy);

		within_width8(y, index8, stride, width);
		*(uchar8*)(Y + index8) = y;
	}
}

#endif


//common
#ifndef IMG_LINEAR_DUAL_2D_ROW_KERNEL4
#define IMG_LINEAR_DUAL_2D_ROW_KERNEL4

//for each row[i]: Y[i] = alpha*X1[i] + beta*X2 + gamma
//(5, 3): Size = 16.0078, Time = 0.076 mesc, Speed = 205.693 GB/s
//(5, 2): Size = 16.0078, Time = 0.09  mesc, Speed = 173.696 GB/s
__global__ void img_linear_dual2D_row_kernel_4(
	const char*  __restrict__ X1,
	const float* __restrict__ X2, int row_lengthv,
	float alpha, float beta, float gamma,
	char* __restrict__ Y,
	int lengthv, int width, int stride)
{
	int step = gridDim.x*blockDim.x, step4 = step << 2;
	int index = blockIdx.x*blockDim.x + threadIdx.x;

	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		uchar4 x1 = *(uchar4*)(X1 + index4);

		int field_index4 = index4 % row_lengthv;//X2_lengthv % 4 == 0
		float4 x2 = *(float4*)(X2 + field_index4);

		float4 fy;//y = alpha*x1 + beta*x2 + gamma
		fy.x = alpha * x1.x + beta * x2.x + gamma;
		fy.y = alpha * x1.y + beta * x2.y + gamma;
		fy.z = alpha * x1.z + beta * x2.z + gamma;
		fy.w = alpha * x1.w + beta * x2.w + gamma;
		uchar4 y; PIXEL_CLIP_4(y, fy);

		within_width4(y, index4, stride, width);
		*(uchar4*)(Y + index4) = y;
	}
}

#endif


void __img_linear_dual2D_row(cudaStream_t stream,
	const char*  X1,
	const float* X2, int row_lengthv,
	float alpha, float beta, float gamma,
	char* Y,
	int lengthv, int width, int stride)
{
	if (lengthv < 256) { img_linear_dual2d_row_k4_small(stream, X1, X2, row_lengthv, alpha, beta, gamma, Y, lengthv, width, stride); return; }
	if (!(lengthv & 15) && lengthv >= 16384) { img_linear_dual2d_row_k16(stream, 5, 4, X1, X2, row_lengthv, alpha, beta, gamma, Y, lengthv, width, stride); return; }
	if (!(lengthv &  7) && lengthv >=  8192) { img_linear_dual2d_row_k8(stream, 5, 3, X1, X2, row_lengthv, alpha, beta, gamma, Y, lengthv, width, stride); return; }
	if (lengthv >= 8192) { img_linear_dual2d_row_k4(stream, 5, 3, X1, X2, row_lengthv, alpha, beta, gamma, Y, lengthv, width, stride); return; }
	img_linear_dual2d_row_k4(stream, 5, 2, X1, X2, row_lengthv, alpha, beta, gamma, Y, lengthv, width, stride);
}

#endif