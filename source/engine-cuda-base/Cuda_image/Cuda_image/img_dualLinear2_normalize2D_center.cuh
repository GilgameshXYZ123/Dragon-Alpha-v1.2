#pragma once

#ifndef IMG_DUAL_LINEAR2_NORMALIZE_2D_CENTER_H
#define IMG_DUAL_LINEAR2_NORMALIZE_2D_CENTER_H

//lengthv = height * stride
//stride = (width + 3)/4*4
//[lenghtv, stride] % 4 == 0
//dim2 % stride == 0 
// X[dim0, dim1, dim2]
//X1[dim0,       dim2]
//X2[dim0,       dim2]
// Y[dim0, dim1, dim2]
#ifndef IMG_DUAL_LINEAR2_NORMALIZE_2D_CENTER_CALL
#define IMG_DUAL_LINEAR2_NORMALIZE_2D_CENTER_CALL

//LB = log2(BLOCK_SIZE)
//LT = log2(THREAD_ELEMENT_VISIT_SIZE)

//lengthv % 16 == 0
#define img_dualLinear2_normalize2d_center_k16(stream, LB, LT, X, X1, X2, Y, dim1, dim2, lengthv, width, stride)\
	img_dualLinear2_normalize2D_center_kernel_16\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(X,X1,X2, alpha1,beta1,gamma1, alpha2,beta2,gamma2, C, Y, (dim1*dim2),dim2, lengthv,width,stride)

//lengthv % 8 == 0
#define img_dualLinear2_normalize2d_center_k8(stream, LB, LT, X, X1, X2, Y, dim1, dim2, lengthv, width, stride)\
	img_dualLinear2_normalize2D_center_kernel_8\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(X,X1,X2, alpha1,beta1,gamma1, alpha2,beta2,gamma2, C, Y, (dim1*dim2),dim2, lengthv,width,stride)

//common
#define img_dualLinear2_normalize2d_center_k4(stream, LB, LT, X, X1, X2, Y, dim1, dim2, lengthv, width, stride)\
	img_dualLinear2_normalize2D_center_kernel_4\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(X,X1,X2, alpha1,beta1,gamma1, alpha2,beta2,gamma2, C, Y, (dim1*dim2),dim2, lengthv,width,stride)

//common
#define img_dualLinear2_normalize2d_center_k4_small(stream, X, X1, X2, Y, dim1, dim2, lengthv, width, stride)\
	img_dualLinear2_normalize2D_center_kernel_4\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(X,X1,X2, alpha1,beta1,gamma1, alpha2,beta2,gamma2, C, Y, (dim1*dim2),dim2, lengthv,width,stride)

#endif


//dim2 % 16 == 0
#ifndef IMG_DUAL_LINEAR2_NORMALIZE_2D_CENTER_KERNEL16
#define IMG_DUAL_LINEAR2_NORMALIZE_2D_CENTER_KERNEL16

//<1> Y1 = alpha1*X  + beta1*X1 + gamma1
//<2> Y2 = alpha2*X1 + beta2*X2 + gamma2
//<3> Y = (Y1 / Y2) + C
__global__ void img_dualLinear2_normalize2D_center_kernel_16(
	const char*  __restrict__ X,
	const float* __restrict__ X1,
	const float* __restrict__ X2,
	float alpha1, float beta1, float gamma1,
	float alpha2, float beta2, float gamma2,
	float C,
	float* __restrict__ Y,
	int dim12, int dim2,//dim12 = dim1 * dim2
	int lengthv, int width, int stride)
{
	int step = gridDim.x*blockDim.x, step16 = step << 4;
	int index = blockIdx.x*blockDim.x + threadIdx.x;

	float4 table[2]; table[0] = F32_0_4;
	for (int index16 = index << 4; index16 < lengthv; index16 += step16)
	{
		uchar16 x = *(uchar16*)(X + index16);
		float4 fy, x1, x2;//<3> Y = (Y1 / Y2) + C
		int d0, d1, d2, r, center_offset, yindex16 = index16;

		//group0: 0-3 items
		d0 = yindex16 / dim12, r = yindex16 - dim12 * d0;//[d0, d1, d2]
		d1 = r / dim2, d2 = r - dim2 * d1;//d2 = index4 % dim2 => {x2[0-3]}
		center_offset = d0 * dim2 + d2;//[d0, d2]
		x1 = *(float4*)(X1 + center_offset);//<1> Y1 = alpha1*X  + beta1*X1 + gamma1
		x2 = *(float4*)(X2 + center_offset);//<2> Y2 = alpha2*X1 + beta2*X2 + gamma2
		fy.x = (alpha1 * x.x0 + beta1 * x1.x + gamma1) / (alpha2 * x1.x + beta2 * x2.x + gamma2) + C;
		fy.y = (alpha1 * x.y0 + beta1 * x1.y + gamma1) / (alpha2 * x1.y + beta2 * x2.y + gamma2) + C;
		fy.z = (alpha1 * x.z0 + beta1 * x1.z + gamma1) / (alpha2 * x1.z + beta2 * x2.z + gamma2) + C;
		fy.w = (alpha1 * x.w0 + beta1 * x1.w + gamma1) / (alpha2 * x1.w + beta2 * x2.w + gamma2) + C;
		within_width4_zero_nan(fy, yindex16, table, stride, width);
		*(float4*)(Y + yindex16) = fy; yindex16 += 4;

		//group1: 4-7 items
		d0 = yindex16 / dim12, r = yindex16 - dim12 * d0;//[d0, d1, d2]
		d1 = r / dim2, d2 = r - dim2 * d1;//d2 = index4 % dim2 => {x2[0-3]}
		center_offset = d0 * dim2 + d2;//[d0, d2]
		x1 = *(float4*)(X1 + center_offset);//<1> Y1 = alpha1*X  + beta1*X1 + gamma1
		x2 = *(float4*)(X2 + center_offset);//<2> Y2 = alpha2*X1 + beta2*X2 + gamma2
		fy.x = (alpha1 * x.x1 + beta1 * x1.x + gamma1) / (alpha2 * x1.x + beta2 * x2.x + gamma2) + C;
		fy.y = (alpha1 * x.y1 + beta1 * x1.y + gamma1) / (alpha2 * x1.y + beta2 * x2.y + gamma2) + C;
		fy.z = (alpha1 * x.z1 + beta1 * x1.z + gamma1) / (alpha2 * x1.z + beta2 * x2.z + gamma2) + C;
		fy.w = (alpha1 * x.w1 + beta1 * x1.w + gamma1) / (alpha2 * x1.w + beta2 * x2.w + gamma2) + C;
		within_width4_zero_nan(fy, yindex16, table, stride, width);
		*(float4*)(Y + yindex16) = fy; yindex16 += 4;

		//group1: 8-11 items
		d0 = yindex16 / dim12, r = yindex16 - dim12 * d0;//[d0, d1, d2]
		d1 = r / dim2, d2 = r - dim2 * d1;//d2 = index4 % dim2 => {x2[0-3]}
		center_offset = d0 * dim2 + d2;//[d0, d2]
		x1 = *(float4*)(X1 + center_offset);//<1> Y1 = alpha1*X  + beta1*X1 + gamma1
		x2 = *(float4*)(X2 + center_offset);//<2> Y2 = alpha2*X1 + beta2*X2 + gamma2
		fy.x = (alpha1 * x.x2 + beta1 * x1.x + gamma1) / (alpha2 * x1.x + beta2 * x2.x + gamma2) + C;
		fy.y = (alpha1 * x.y2 + beta1 * x1.y + gamma1) / (alpha2 * x1.y + beta2 * x2.y + gamma2) + C;
		fy.z = (alpha1 * x.z2 + beta1 * x1.z + gamma1) / (alpha2 * x1.z + beta2 * x2.z + gamma2) + C;
		fy.w = (alpha1 * x.w2 + beta1 * x1.w + gamma1) / (alpha2 * x1.w + beta2 * x2.w + gamma2) + C;
		within_width4_zero_nan(fy, yindex16, table, stride, width);
		*(float4*)(Y + yindex16) = fy; yindex16 += 4;

		//group1: 12-15 items
		d0 = yindex16 / dim12, r = yindex16 - dim12 * d0;//[d0, d1, d2]
		d1 = r / dim2, d2 = r - dim2 * d1;//d2 = index4 % dim2 => {x2[0-3]}
		center_offset = d0 * dim2 + d2;//[d0, d2]
		x1 = *(float4*)(X1 + center_offset);//<1> Y1 = alpha1*X  + beta1*X1 + gamma1
		x2 = *(float4*)(X2 + center_offset);//<2> Y2 = alpha2*X1 + beta2*X2 + gamma2
		fy.x = (alpha1 * x.x3 + beta1 * x1.x + gamma1) / (alpha2 * x1.x + beta2 * x2.x + gamma2) + C;
		fy.y = (alpha1 * x.y3 + beta1 * x1.y + gamma1) / (alpha2 * x1.y + beta2 * x2.y + gamma2) + C;
		fy.z = (alpha1 * x.z3 + beta1 * x1.z + gamma1) / (alpha2 * x1.z + beta2 * x2.z + gamma2) + C;
		fy.w = (alpha1 * x.w3 + beta1 * x1.w + gamma1) / (alpha2 * x1.w + beta2 * x2.w + gamma2) + C;
		within_width4_zero_nan(fy, yindex16, table, stride, width);
		*(float4*)(Y + yindex16) = fy;
	}
}

#endif


//dim2 % 8 == 0
#ifndef IMG_DUAL_LINEAR2_NORMALIZE_2D_CENTER_KERNEL8
#define IMG_DUAL_LINEAR2_NORMALIZE_2D_CENTER_KERNEL8

//<1> Y1 = alpha1*X  + beta1*X1 + gamma1
//<2> Y2 = alpha2*X1 + beta2*X2 + gamma2
//<3> Y = (Y1 / Y2) + C
__global__ void img_dualLinear2_normalize2D_center_kernel_8(
	const char*  __restrict__ X,
	const float* __restrict__ X1,
	const float* __restrict__ X2,
	float alpha1, float beta1, float gamma1,
	float alpha2, float beta2, float gamma2,
	float C,
	float* __restrict__ Y,
	int dim12, int dim2,//dim12 = dim1 * dim2
	int lengthv, int width, int stride)
{
	int step = gridDim.x*blockDim.x, step8 = step << 3;
	int index = blockIdx.x*blockDim.x + threadIdx.x;

	float4 table[2]; table[0] = F32_0_4;
	for (int index8 = index << 3; index8 < lengthv; index8 += step8)
	{
		uchar8 x = *(uchar8*)(X + index8);
		float4 fy, x1, x2;//<3> Y = (Y1 / Y2) + C
		int d0, d1, d2, r, center_offset, yindex8 = index8;

		//group0: 0-3 items
		d0 = yindex8 / dim12, r = yindex8 - dim12 * d0;//[d0, d1, d2]
		d1 = r / dim2, d2 = r - dim2 * d1;//d2 = index4 % dim2 => {x2[0-3]}
		center_offset = d0 * dim2 + d2;//[d0, d2]
		x1 = *(float4*)(X1 + center_offset);//<1> Y1 = alpha1*X  + beta1*X1 + gamma1
		x2 = *(float4*)(X2 + center_offset);//<2> Y2 = alpha2*X1 + beta2*X2 + gamma2

		fy.x = (alpha1 * x.x0 + beta1 * x1.x + gamma1) / (alpha2 * x1.x + beta2 * x2.x + gamma2) + C;
		fy.y = (alpha1 * x.y0 + beta1 * x1.y + gamma1) / (alpha2 * x1.y + beta2 * x2.y + gamma2) + C;
		fy.z = (alpha1 * x.z0 + beta1 * x1.z + gamma1) / (alpha2 * x1.z + beta2 * x2.z + gamma2) + C;
		fy.w = (alpha1 * x.w0 + beta1 * x1.w + gamma1) / (alpha2 * x1.w + beta2 * x2.w + gamma2) + C;
		within_width4_zero_nan(fy, yindex8, table, stride, width);
		*(float4*)(Y + yindex8) = fy; yindex8 += 4;

		//group1: 4-7 items
		d0 = yindex8 / dim12, r = yindex8 - dim12 * d0;//[d0, d1, d2]
		d1 = r / dim2, d2 = r - dim2 * d1;//d2 = index4 % dim2 => {x2[0-3]}
		center_offset = d0 * dim2 + d2;//[d0, d2]
		x1 = *(float4*)(X1 + center_offset);//<1> Y1 = alpha1*X  + beta1*X1 + gamma1
		x2 = *(float4*)(X2 + center_offset);//<2> Y2 = alpha2*X1 + beta2*X2 + gamma2

		fy.x = (alpha1 * x.x1 + beta1 * x1.x + gamma1) / (alpha2 * x1.x + beta2 * x2.x + gamma2) + C;
		fy.y = (alpha1 * x.y1 + beta1 * x1.y + gamma1) / (alpha2 * x1.y + beta2 * x2.y + gamma2) + C;
		fy.z = (alpha1 * x.z1 + beta1 * x1.z + gamma1) / (alpha2 * x1.z + beta2 * x2.z + gamma2) + C;
		fy.w = (alpha1 * x.w1 + beta1 * x1.w + gamma1) / (alpha2 * x1.w + beta2 * x2.w + gamma2) + C;
		within_width4_zero_nan(fy, yindex8, table, stride, width);
		*(float4*)(Y + yindex8) = fy;
	}
}

#endif


//common
#ifndef IMG_DUAL_LINEAR2_NORMALIZE_2D_CENTER_KERNEL4
#define IMG_DUAL_LINEAR2_NORMALIZE_2D_CENTER_KERNEL4

//<1> Y1 = alpha1*X  + beta1*X1 + gamma1
//<2> Y2 = alpha2*X1 + beta2*X2 + gamma2
//<3> Y = (Y1 / Y2) + C
__global__ void img_dualLinear2_normalize2D_center_kernel_4(
	const char*  __restrict__ X,
	const float* __restrict__ X1,
	const float* __restrict__ X2,
	float alpha1, float beta1, float gamma1,
	float alpha2, float beta2, float gamma2,
	float C,
	float* __restrict__ Y,
	int dim12, int dim2,//dim12 = dim1 * dim2
	int lengthv, int width, int stride)
{
	int step = gridDim.x*blockDim.x, step4 = step << 2;
	int index = blockIdx.x*blockDim.x + threadIdx.x;

	float4 table[2]; table[0] = F32_0_4;
	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		uchar4 x = *(uchar4*)(X + index4);
		float4 fy, x1, x2;//<3> Y = Y1 / (alpha2*X2 + beta2) + C

		//group0: 0-3 items
		int d0 = index4 / dim12, r = index4 - dim12 * d0;//[d0, d1, d2]
		int d1 = r / dim2, d2 = r - dim2 * d1;//d2 = index4 % dim2 => {x2[0-3]}
		int center_offset = d0 * dim2 + d2;//[d0, d2]

		x1 = *(float4*)(X1 + center_offset);//<1> Y1 = alpha1*X  + beta1*X1 + gamma1
		x2 = *(float4*)(X2 + center_offset);//<2> Y2 = alpha2*X1 + beta2*X2 + gamma2

		fy.x = (alpha1 * x.x + beta1 * x1.x + gamma1) / (alpha2 * x1.x + beta2 * x2.x + gamma2) + C;
		fy.y = (alpha1 * x.y + beta1 * x1.y + gamma1) / (alpha2 * x1.y + beta2 * x2.y + gamma2) + C;
		fy.z = (alpha1 * x.z + beta1 * x1.z + gamma1) / (alpha2 * x1.z + beta2 * x2.z + gamma2) + C;
		fy.w = (alpha1 * x.w + beta1 * x1.w + gamma1) / (alpha2 * x1.w + beta2 * x2.w + gamma2) + C;
		within_width4_zero_nan(fy, index4, table, stride, width);
		*(float4*)(Y + index4) = fy;
	}
}

#endif


void __img_dualLinear2_normalize2D_center(cudaStream_t stream,
	const char* X,
	const float* X1, const float* X2,
	float alpha1, float beta1, float gamma1,
	float alpha2, float beta2, float gamma2,
	float C,
	float *Y,
	int dim0, int dim1, int dim2,
	int width, int stride)
{
	int lengthv = dim0 * dim1 * dim2;//dim2 % 4 == 0
	if (lengthv < 256) { img_dualLinear2_normalize2d_center_k4_small(stream, X, X1, X2, Y, dim1, dim2, lengthv, width, stride); return; }
	if (!(dim2 & 15) && lengthv >= 16384) { img_dualLinear2_normalize2d_center_k16(stream, 5, 4, X, X1, X2, Y, dim1, dim2, lengthv, width, stride); return; }
	if (!(dim2 &  7) && lengthv >=  8192) { img_dualLinear2_normalize2d_center_k8 (stream, 5, 3, X, X1, X2, Y, dim1, dim2, lengthv, width, stride); return; }
	if (lengthv >= 8192) { img_dualLinear2_normalize2d_center_k4(stream, 5, 3, X, X1, X2, Y, dim1, dim2, lengthv, width, stride); return; }
	img_dualLinear2_normalize2d_center_k4(stream, 5, 2, X, X1, X2, Y, dim1, dim2, lengthv, width, stride);
}

#endif