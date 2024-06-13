#pragma once

#ifndef IMG_LINEAR_DUAL_2D_FIELD_H
#define IMG_LINEAR_DUAL_2D_FIELD_H

//lengthv = height * stride
//stride = (width + 3)/4 * 4, stride >= 4, stride % 4 ==0
//[lengthv, row_lengthv] % stride == 0
//X2 must a 1D Tensor[field_length]
//field_length * row_lengthv = X1.lengthv
#ifndef IMG_LINEAR_DUAL_2D_FIELD_CALL
#define IMG_LINEAR_DUAL_2D_FIELD_CALL

//LB = log2(BLOCK_SIZE)
//LT = log2(THREAD_ELEMENT_VISIT_SIZE)

//lengthv % 16 == 0
#define img_linear_dual2d_field_k16(stream, LB, LT, X1, X2, row_lengthv, alpha, beta, gamma, Y, lengthv, width, stride)\
	img_linear_dual2D_field_kernel_16\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(X1, X2, row_lengthv, alpha, beta, gamma, Y, lengthv, width, stride)

//lengthv % 8 == 0
#define img_linear_dual2d_field_k8(stream, LB, LT, X1, X2, row_lengthv, alpha, beta, gamma, Y, lengthv, width, stride)\
	img_linear_dual2D_field_kernel_8\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(X1, X2, row_lengthv, alpha, beta, gamma, Y, lengthv, width, stride)

//common
#define img_linear_dual2d_field_k4(stream, LB, LT, X1, X2, row_lengthv, alpha, beta, gamma, Y, lengthv, width, stride)\
	img_linear_dual2D_field_kernel_4\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(X1, X2, row_lengthv, alpha, beta, gamma, Y, lengthv, width, stride)

//common
#define img_linear_dual2d_field_k4_small(stream, X1, X2, row_lengthv, alpha, beta, gamma, Y, lengthv, width, stride)\
	img_linear_dual2D_field_kernel_4\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(X1, X2, row_lengthv, alpha, beta, gamma, Y, lengthv, width, stride)

#endif


//lengthv % 16 == 0
#ifndef IMG_LINEAR_DUAL_2D_FIELD_KERNEL16
#define IMG_LINEAR_DUAL_2D_FIELD_KERNEL16

//for each field[i]: Y[i] = alpha*X1[i] + beta*X2 + gamma
//(5, 4): Size = 16, Time = 0.055 mesc, Speed = 284.091 GB/s
__global__ void img_linear_dual2D_field_kernel_16(
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
		float C0 = X2[(index16     ) / row_lengthv] * beta + gamma;
		float C1 = X2[(index16 +  4) / row_lengthv] * beta + gamma;
		float C2 = X2[(index16 +  8) / row_lengthv] * beta + gamma;
		float C3 = X2[(index16 + 12) / row_lengthv] * beta + gamma;

		float16 fy;//y = alpha*x + beta
		fy.x0 = (alpha * x1.x0) + C0;
		fy.y0 = (alpha * x1.y0) + C0;
		fy.z0 = (alpha * x1.z0) + C0;
		fy.w0 = (alpha * x1.w0) + C0;

		fy.x1 = (alpha * x1.x1) + C1;
		fy.y1 = (alpha * x1.y1) + C1;
		fy.z1 = (alpha * x1.z1) + C1;
		fy.w1 = (alpha * x1.w1) + C1;

		fy.x2 = (alpha * x1.x2) + C2;
		fy.y2 = (alpha * x1.y2) + C2;
		fy.z2 = (alpha * x1.z2) + C2;
		fy.w2 = (alpha * x1.w2) + C2;

		fy.x3 = (alpha * x1.x3) + C3;
		fy.y3 = (alpha * x1.y3) + C3;
		fy.z3 = (alpha * x1.z3) + C3;
		fy.w3 = (alpha * x1.w3) + C3;
		uchar16 y; PIXEL_CLIP_16(y, fy);

		within_width16(y, index16, stride, width);
		*(uchar16*)(Y + index16) = y;
	}
}

#endif


//lengthv % 8 == 0
#ifndef IMG_LINEAR_DUAL_2D_FIELD_KERNEL8
#define IMG_LINEAR_DUAL_2D_FIELD_KERNEL8

//for each field[i]: Y[i] = alpha*X1[i] + beta*X2 + gamma
//(5, 3): Size = 16, Time = 0.066 mesc, Speed = 236.742 GB/s
__global__ void img_linear_dual2D_field_kernel_8(
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
		float C0 = X2[(index8    ) / row_lengthv] * beta + gamma;
		float C1 = X2[(index8 + 4) / row_lengthv] * beta + gamma;

		float8 fy;//y = alpha*x1 + beta*x2 + gamma
		fy.x0 = (alpha * x1.x0) + C0;
		fy.y0 = (alpha * x1.y0) + C0;
		fy.z0 = (alpha * x1.z0) + C0;
		fy.w0 = (alpha * x1.w0) + C0;

		fy.x1 = (alpha * x1.x1) + C1;
		fy.y1 = (alpha * x1.y1) + C1;
		fy.z1 = (alpha * x1.z1) + C1;
		fy.w1 = (alpha * x1.w1) + C1;
		uchar8 y; PIXEL_CLIP_8(y, fy);

		within_width8(y, index8, stride, width);
		*(uchar8*)(Y + index8) = y;
	}
}

#endif


//common
#ifndef IMG_LINEAR_DUAL_2D_FIELD_KERNEL4
#define IMG_LINEAR_DUAL_2D_FIELD_KERNEL4

//for each field[i]: Y[i] = alpha*X1[i] + beta*X2 + gamma
//(5, 3): Size = 16, Time = 0.082 mesc, Speed = 190.549 GB/s
//(5, 2): Size = 16, Time = 0.094 mesc, Speed = 166.223 GB/s
__global__ void img_linear_dual2D_field_kernel_4(
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
		float C = X2[index4 / row_lengthv] * beta + gamma;

		float4 fy;//y = alpha*x1 + beta*x2 + gamma
		fy.x = (alpha * x1.x) + C;
		fy.y = (alpha * x1.y) + C;
		fy.z = (alpha * x1.z) + C;
		fy.w = (alpha * x1.w) + C;
		uchar4 y; PIXEL_CLIP_4(y, fy);

		within_width4(y, index4, stride, width);
		*(uchar4*)(Y + index4) = y;
	}
}

#endif


void __img_linear_dual2D_field(cudaStream_t stream,
	const char*  X1,
	const float* X2, int row_lengthv,
	float alpha, float beta, float gamma,
	char* Y,
	int lengthv, int width, int stride)
{
	if (lengthv < 256) { img_linear_dual2d_field_k4_small(stream, X1, X2, row_lengthv, alpha, beta, gamma, Y, lengthv, width, stride); return; }
	if (!(lengthv & 15) && lengthv >= 16384) { img_linear_dual2d_field_k16(stream, 5, 4, X1, X2, row_lengthv, alpha, beta, gamma, Y, lengthv, width, stride); return; }
	if (!(lengthv &  7) && lengthv >=  8192) { img_linear_dual2d_field_k8(stream, 5, 3, X1, X2, row_lengthv, alpha, beta, gamma, Y, lengthv, width, stride); return; }
	if (lengthv >= 8192) { img_linear_dual2d_field_k4(stream, 5, 3, X1, X2, row_lengthv, alpha, beta, gamma, Y, lengthv, width, stride); return; }
	img_linear_dual2d_field_k4(stream, 5, 2, X1, X2, row_lengthv, alpha, beta, gamma, Y, lengthv, width, stride);
}

#endif