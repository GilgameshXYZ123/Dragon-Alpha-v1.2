#pragma once

#ifndef IMG_DUAL_LINEAR2_DIVIDE_2D_H
#define IMG_DUAL_LINEAR2_DIVIDE_2D_H

//lengthv = height * stride
//stride = (width + 3)/4*4
//[lenghtv, stride] % 4 == 0
#ifndef IMG_DUAL_LINEAR2_DIVIDE_2D_CALL
#define IMG_DUAL_LINEAR2_DIVIDE_2D_CALL

//LB = log2(BLOCK_SIZE)
//LT = log2(THREAD_ELEMENT_VISIT_SIZE)

//lengthv % 16 == 0
#define img_dualLinear2_divide2d_k16(stream, LB, LT, X, X1, X2, Y, lengthv, width, stride)\
	img_dualLinear2_divide2D_kernel_16\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(X,X1,X2, alpha1,beta1,gamma1, alpha2,beta2,gamma2, C, Y, lengthv,width,stride)

//lengthv % 8 == 0
#define img_dualLinear2_divide2d_k8(stream, LB, LT, X, X1, X2, Y, lengthv, width, stride)\
	img_dualLinear2_divide2D_kernel_8\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(X,X1,X2, alpha1,beta1,gamma1, alpha2,beta2,gamma2, C, Y, lengthv,width,stride)

//common
#define img_dualLinear2_divide2d_k4(stream, LB, LT, X, X1, X2, Y, lengthv, width, stride)\
	img_dualLinear2_divide2D_kernel_4\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(X,X1,X2, alpha1,beta1,gamma1, alpha2,beta2,gamma2, C, Y, lengthv,width,stride)

//common
#define img_dualLinear2_divide2d_k4_small(stream, X, X1, X2, Y, lengthv, width, stride)\
	img_dualLinear2_divide2D_kernel_4\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(X,X1,X2, alpha1,beta1,gamma1, alpha2,beta2,gamma2, C, Y, lengthv,width,stride)

#endif


//lengthv % 16 == 0
#ifndef IMG_DUAL_LINEAR2_DIVIDE_2D_KERNEL16
#define IMG_DUAL_LINEAR2_DIVIDE_2D_KERNEL16

//<1> Y1 = alpha1*X + beta1*X1 + gamma1
//<2> Y2 = alpha2*X + beta2*X2 + gamma2
//<3> Y = (Y1 / Y2) + C
//(5, 4): Size = 56, Time = 0.147 mesc, Speed = 372.024 GB/s
__global__ void img_dualLinear2_divide2D_kernel_16(
	const char* __restrict__ X,
	const char* __restrict__ X1,
	const char* __restrict__ X2,
	float alpha1, float beta1, float gamma1,
	float alpha2, float beta2, float gamma2,
	float C,
	float* __restrict__ Y,
	int lengthv, int width, int stride)
{
	int step = gridDim.x*blockDim.x, step16 = step << 4;
	int index = blockIdx.x*blockDim.x + threadIdx.x;

	float4 table[2]; table[0] = F32_0_4;
	for (int index16 = index << 4; index16 < lengthv; index16 += step16)
	{
		uchar16 x  = *(uchar16*)(X  + index16);
		uchar16 x1 = *(uchar16*)(X1 + index16);
		uchar16 x2 = *(uchar16*)(X2 + index16);

		float16 fy1;//<1> Y1 = alpha1*X + beta1*X1 + gamma1
		fy1.x0 = alpha1 * x.x0 + beta1 * x1.x0 + gamma1;
		fy1.y0 = alpha1 * x.y0 + beta1 * x1.y0 + gamma1;
		fy1.z0 = alpha1 * x.z0 + beta1 * x1.z0 + gamma1;
		fy1.w0 = alpha1 * x.w0 + beta1 * x1.w0 + gamma1;

		fy1.x1 = alpha1 * x.x1 + beta1 * x1.x1 + gamma1;
		fy1.y1 = alpha1 * x.y1 + beta1 * x1.y1 + gamma1;
		fy1.z1 = alpha1 * x.z1 + beta1 * x1.z1 + gamma1;
		fy1.w1 = alpha1 * x.w1 + beta1 * x1.w1 + gamma1;

		fy1.x2 = alpha1 * x.x2 + beta1 * x1.x2 + gamma1;
		fy1.y2 = alpha1 * x.y2 + beta1 * x1.y2 + gamma1;
		fy1.z2 = alpha1 * x.z2 + beta1 * x1.z2 + gamma1;
		fy1.w2 = alpha1 * x.w2 + beta1 * x1.w2 + gamma1;

		fy1.x3 = alpha1 * x.x3 + beta1 * x1.x3 + gamma1;
		fy1.y3 = alpha1 * x.y3 + beta1 * x1.y3 + gamma1;
		fy1.z3 = alpha1 * x.z3 + beta1 * x1.z3 + gamma1;
		fy1.w3 = alpha1 * x.w3 + beta1 * x1.w3 + gamma1;

		float16 fy2;//<2> Y2 = alpha2*X + beta2*X2 + gamma2
		fy2.x0 = alpha2 * x.x0 + beta2 * x2.x0 + gamma2;
		fy2.y0 = alpha2 * x.y0 + beta2 * x2.y0 + gamma2;
		fy2.z0 = alpha2 * x.z0 + beta2 * x2.z0 + gamma2;
		fy2.w0 = alpha2 * x.w0 + beta2 * x2.w0 + gamma2;

		fy2.x1 = alpha2 * x.x1 + beta2 * x2.x1 + gamma2;
		fy2.y1 = alpha2 * x.y1 + beta2 * x2.y1 + gamma2;
		fy2.z1 = alpha2 * x.z1 + beta2 * x2.z1 + gamma2;
		fy2.w1 = alpha2 * x.w1 + beta2 * x2.w1 + gamma2;

		fy2.x2 = alpha2 * x.x2 + beta2 * x2.x2 + gamma2;
		fy2.y2 = alpha2 * x.y2 + beta2 * x2.y2 + gamma2;
		fy2.z2 = alpha2 * x.z2 + beta2 * x2.z2 + gamma2;
		fy2.w2 = alpha2 * x.w2 + beta2 * x2.w2 + gamma2;

		fy2.x3 = alpha2 * x.x3 + beta2 * x2.x3 + gamma2;
		fy2.y3 = alpha2 * x.y3 + beta2 * x2.y3 + gamma2;
		fy2.z3 = alpha2 * x.z3 + beta2 * x2.z3 + gamma2;
		fy2.w3 = alpha2 * x.w3 + beta2 * x2.w3 + gamma2;

		int yindex16 = index16;
		float4 fy;//<3> Y = (Y1 / Y2) + C
		fy.x = (fy1.x0 / fy2.x0) + C;
		fy.y = (fy1.y0 / fy2.y0) + C;
		fy.z = (fy1.z0 / fy2.z0) + C;
		fy.w = (fy1.w0 / fy2.w0) + C;

		within_width4_zero_nan(fy, yindex16, table, stride, width); 
		*(float4*)(Y + yindex16) = fy; yindex16 += 4;

		fy.x = (fy1.x1 / fy2.x1) + C;
		fy.y = (fy1.y1 / fy2.y1) + C;
		fy.z = (fy1.z1 / fy2.z1) + C;
		fy.w = (fy1.w1 / fy2.w1) + C;

		within_width4_zero_nan(fy, yindex16, table, stride, width);
		*(float4*)(Y + yindex16) = fy; yindex16 += 4;

		fy.x = (fy1.x2 / fy2.x2) + C;
		fy.y = (fy1.y2 / fy2.y2) + C;
		fy.z = (fy1.z2 / fy2.z2) + C;
		fy.w = (fy1.w2 / fy2.w2) + C;

		within_width4_zero_nan(fy, yindex16, table, stride, width);
		*(float4*)(Y + yindex16) = fy; yindex16 += 4;

		fy.x = (fy1.x3 / fy2.x3) + C;
		fy.y = (fy1.y3 / fy2.y3) + C;
		fy.z = (fy1.z3 / fy2.z3) + C;
		fy.w = (fy1.w3 / fy2.w3) + C;

		within_width4_zero_nan(fy, yindex16, table, stride, width);
		*(float4*)(Y + yindex16) = fy; 
	}
}

#endif


//lengthv % 8 == 0
#ifndef IMG_DUAL_LINEAR2_DIVIDE_2D_KERNEL8
#define IMG_DUAL_LINEAR2_DIVIDE_2D_KERNEL8

//<1> Y1 = alpha1*X + beta1*X1 + gamma1
//<2> Y2 = alpha2*X + beta2*X2 + gamma2
//<3> Y = (Y1 / Y2) + C
//(5, 3): Size = 56, Time = 0.148 mesc, Speed = 369.51 GB/s
__global__ void img_dualLinear2_divide2D_kernel_8(
	const char* __restrict__ X,
	const char* __restrict__ X1,
	const char* __restrict__ X2,
	float alpha1, float beta1, float gamma1,
	float alpha2, float beta2, float gamma2,
	float C,
	float* __restrict__ Y,
	int lengthv, int width, int stride)
{
	int step = gridDim.x*blockDim.x, step8 = step << 3;
	int index = blockIdx.x*blockDim.x + threadIdx.x;

	float4 table[2]; table[0] = F32_0_4;
	for (int index8 = index << 3; index8 < lengthv; index8 += step8)
	{
		uchar8 x  = *(uchar8*)(X  + index8);
		uchar8 x1 = *(uchar8*)(X1 + index8);
		uchar8 x2 = *(uchar8*)(X2 + index8);

		float8 fy1;//<1> Y1 = alpha1*X + beta1*X1 + gamma1
		fy1.x0 = alpha1 * x.x0 + beta1 * x1.x0 + gamma1;
		fy1.y0 = alpha1 * x.y0 + beta1 * x1.y0 + gamma1;
		fy1.z0 = alpha1 * x.z0 + beta1 * x1.z0 + gamma1;
		fy1.w0 = alpha1 * x.w0 + beta1 * x1.w0 + gamma1;

		fy1.x1 = alpha1 * x.x1 + beta1 * x1.x1 + gamma1;
		fy1.y1 = alpha1 * x.y1 + beta1 * x1.y1 + gamma1;
		fy1.z1 = alpha1 * x.z1 + beta1 * x1.z1 + gamma1;
		fy1.w1 = alpha1 * x.w1 + beta1 * x1.w1 + gamma1;

		float8 fy2;//<2> Y2 = alpha2*X + beta2*X2 + gamma2
		fy2.x0 = alpha2 * x.x0 + beta2 * x2.x0 + gamma2;
		fy2.y0 = alpha2 * x.y0 + beta2 * x2.y0 + gamma2;
		fy2.z0 = alpha2 * x.z0 + beta2 * x2.z0 + gamma2;
		fy2.w0 = alpha2 * x.w0 + beta2 * x2.w0 + gamma2;

		fy2.x1 = alpha2 * x.x1 + beta2 * x2.x1 + gamma2;
		fy2.y1 = alpha2 * x.y1 + beta2 * x2.y1 + gamma2;
		fy2.z1 = alpha2 * x.z1 + beta2 * x2.z1 + gamma2;
		fy2.w1 = alpha2 * x.w1 + beta2 * x2.w1 + gamma2;

		int yindex8 = index8;
		float4 fy;//<3> Y = (Y1 / Y2) + C
		fy.x = (fy1.x0 / fy2.x0) + C;
		fy.y = (fy1.y0 / fy2.y0) + C;
		fy.z = (fy1.z0 / fy2.z0) + C;
		fy.w = (fy1.w0 / fy2.w0) + C;

		within_width4_zero_nan(fy, yindex8, table, stride, width);
		*(float4*)(Y + yindex8) = fy; yindex8 += 4;

		fy.x = (fy1.x1 / fy2.x1) + C;
		fy.y = (fy1.y1 / fy2.y1) + C;
		fy.z = (fy1.z1 / fy2.z1) + C;
		fy.w = (fy1.w1 / fy2.w1) + C;

		within_width4_zero_nan(fy, yindex8, table, stride, width);
		*(float4*)(Y + yindex8) = fy;
		
	}
}

#endif


//common
#ifndef IMG_DUAL_LINEAR2_DIVIDE_2D_KERNEL4
#define IMG_DUAL_LINEAR2_DIVIDE_2D_KERNEL4

//<1> Y1 = alpha1*X + beta1*X1 + gamma1
//<2> Y2 = alpha2*X + beta2*X2 + gamma2
//<3> Y = (Y1 / Y2) + C
//(5, 2): Size = 56, Time = 0.155 mesc, Speed = 352.823 GB/s
//(5, 3): Size = 56, Time = 0.152 mesc, Speed = 359.786 GB/s
__global__ void img_dualLinear2_divide2D_kernel_4(
	const char* __restrict__ X,
	const char* __restrict__ X1,
	const char* __restrict__ X2,
	float alpha1, float beta1, float gamma1,
	float alpha2, float beta2, float gamma2,
	float C,
	float* __restrict__ Y,
	int lengthv, int width, int stride)
{
	int step = gridDim.x*blockDim.x, step4 = step << 2;
	int index = blockIdx.x*blockDim.x + threadIdx.x;

	float4 table[2]; table[0] = F32_0_4;
	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		uchar4 x  = *(uchar4*)(X  + index4);
		uchar4 x1 = *(uchar4*)(X1 + index4);
		uchar4 x2 = *(uchar4*)(X2 + index4);

		float4 fy1;//<1> Y1 = alpha1*X + beta1*X1 + gamma1
		fy1.x = alpha1 * x.x + beta1 * x1.x + gamma1;
		fy1.y = alpha1 * x.y + beta1 * x1.y + gamma1;
		fy1.z = alpha1 * x.z + beta1 * x1.z + gamma1;
		fy1.w = alpha1 * x.w + beta1 * x1.w + gamma1;

		float4 fy2;//<2> Y2 = alpha2*X + beta2*X2 + gamma2
		fy2.x = alpha2 * x.x + beta2 * x2.x + gamma2;
		fy2.y = alpha2 * x.y + beta2 * x2.y + gamma2;
		fy2.z = alpha2 * x.z + beta2 * x2.z + gamma2;
		fy2.w = alpha2 * x.w + beta2 * x2.w + gamma2;

		float4 fy;//<3> Y = (Y1 / Y2) + C
		fy.x = (fy1.x / fy2.x) + C;
		fy.y = (fy1.y / fy2.y) + C;
		fy.z = (fy1.z / fy2.z) + C;
		fy.w = (fy1.w / fy2.w) + C;

		within_width4_zero_nan(fy, index4, table, stride, width);
		*(float4*)(Y + index4) = fy;
	}
}

#endif


void __img_dualLinear2_div(cudaStream_t stream,
	const char* X, 
	const char* X1, 
	const char* X2,
	float alpha1, float beta1, float gamma1,
	float alpha2, float beta2, float gamma2, 
	float C,
	float *Y, 
	int lengthv, int width, int stride)
{
	if (lengthv < 256) { img_dualLinear2_divide2d_k4_small(stream, X, X1, X2, Y, lengthv, width, stride); return; }
	if (!(lengthv & 15) && lengthv >= 16384) { img_dualLinear2_divide2d_k16(stream, 5, 4, X, X1, X2, Y, lengthv, width, stride); return; }
	if (!(lengthv &  7) && lengthv >=  8192) { img_dualLinear2_divide2d_k8(stream, 5, 3, X, X1, X2, Y, lengthv, width, stride); return; }
	if (lengthv >= 8192) { img_dualLinear2_divide2d_k4(stream, 5, 3, X, X1, X2, Y, lengthv, width, stride); return; }
	img_dualLinear2_divide2d_k4(stream, 5, 2, X, X1, X2, Y, lengthv, width, stride);
}

#endif