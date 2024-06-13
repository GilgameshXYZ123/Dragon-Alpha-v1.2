#pragma once

#ifndef IMG_LINEAR2_DIV_2D_FIELD_H
#define IMG_LINEAR2_DIV_2D_FIELD_H

//lengthv = height * stride
//stride = (width + 3)/4 * 4, stride >= 4, stride % 4 ==0
//[lengthv, row_lengthv] % stride == 0
//X2 must a 1D Tensor[field_length]
//field_length * row_lengthv = X1.lengthv
#ifndef IMG_LINEAR2_DIV_2D_FIELD_CALL
#define IMG_LINEAR2_DIV_2D_FIELD_CALL

//LB = log2(BLOCK_SIZE)
//LT = log2(THREAD_ELEMENT_VISIT_SIZE)

//lengthv % 8 == 0
#define img_linear2_div2d_field_k16(stream, LB, LT, X, X1, X2, row_lengthv, Y, lengthv, width, stride)\
	img_linear2_div2D_field_kernel_16\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(X, X1, X2, row_lengthv, alpha1, beta1, gamma1, alpha2, beta2, C, Y, lengthv, width, stride)

//lengthv % 8 == 0
#define img_linear2_div2d_field_k8(stream, LB, LT, X, X1, X2, row_lengthv, Y, lengthv, width, stride)\
	img_linear2_div2D_field_kernel_8\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(X, X1, X2, row_lengthv, alpha1, beta1, gamma1, alpha2, beta2, C, Y, lengthv, width, stride)

//common
#define img_linear2_div2d_field_k4(stream, LB, LT, X, X1, X2, row_lengthv, Y, lengthv, width, stride)\
	img_linear2_div2D_field_kernel_4\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(X, X1, X2, row_lengthv, alpha1, beta1, gamma1, alpha2, beta2, C, Y, lengthv, width, stride)

//common
#define img_linear2_div2d_field_k4_small(stream, X, X1, X2, row_lengthv, Y, lengthv, width, stride)\
	img_linear2_div2D_field_kernel_4\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(X, X1, X2, row_lengthv, alpha1, beta1, gamma1, alpha2, beta2, C, Y, lengthv, width, stride)

#endif


//lengthv % 16 == 0
#ifndef IMG_LINEAR2_DIV_2D_FIELD_KERNEL16
#define IMG_LINEAR2_DIV_2D_FIELD_KERNEL16

//for each field[i]: 
//	Y[i] = (alpha1*X[i] + beta1*X1 + gamma1) / (alpha2*X2 + beta2) + C
//STEP:
//<1> A = beta1*X1 + gamma1
//<2> B = 1 / (alpha2*X2 + beta2)
//<3> K = alpha1*B
//<4> L = A * B + C
//<5> Y = K * X + L
//
//(5, 4): Size = 40.0156, Time = 0.109 mesc, Speed = 358.512 GB/s
__global__ void img_linear2_div2D_field_kernel_16(
	const char*  __restrict__ X,
	const float* __restrict__ X1,
	const float* __restrict__ X2, int row_lengthv,
	float alpha1, float beta1, float gamma1,
	float alpha2, float beta2, float C,
	float* __restrict__ Y,
	int lengthv, int width, int stride)
{
	int step = gridDim.x*blockDim.x, step16 = step << 4;
	int index = blockIdx.x*blockDim.x + threadIdx.x;

	float4 table[2]; table[0] = F32_0_4;
	for (int index16 = index << 4; index16 < lengthv; index16 += step16)
	{
		uchar16 x = *(uchar16*)(X + index16);

		//row_lengthv % 16 == 0: field_index{0, 1, 2, 3} are the same
		//index16 % 16 == 0, so: index16 = 16x
		//field_index[i] = (index16 + 4*i) / 16 = (16x + 4*i) / 16 = x
		int row_index = index16 / row_lengthv;
		float A = beta1 * X1[row_index] + gamma1;//<1> A = beta1*X1 + gamma1
		float B = 1.0f / (alpha2 * X2[row_index] + beta2);//<2> B = 1 / (alpha2*X2 + beta2)
		float K = alpha1 * B;//<3> K = alpha1*B
		float L = A * B + C;//<4> L = A * B + C

		float4 y; int yindex16 = index16;//<3> Y = (alpha1*X[i] + A)*B + C
		y.x = K * x.x0 + L;
		y.y = K * x.y0 + L;
		y.z = K * x.z0 + L;
		y.w = K * x.w0 + L;
		within_width4_zero_nan(y, yindex16, table, stride, width);
		*(float4*)(Y + yindex16) = y; yindex16 += 4;

		y.x = K * x.x1 + L;
		y.y = K * x.y1 + L;
		y.z = K * x.z1 + L;
		y.w = K * x.w1 + L;
		within_width4_zero_nan(y, yindex16, table, stride, width);
		*(float4*)(Y + yindex16) = y; yindex16 += 4;

		y.x = K * x.x2 + L;
		y.y = K * x.y2 + L;
		y.z = K * x.z2 + L;
		y.w = K * x.w2 + L;
		within_width4_zero_nan(y, yindex16, table, stride, width);
		*(float4*)(Y + yindex16) = y; yindex16 += 4;

		y.x = K * x.x3 + L;
		y.y = K * x.y3 + L;
		y.z = K * x.z3 + L;
		y.w = K * x.w3 + L;
		within_width4_zero_nan(y, yindex16, table, stride, width);
		*(float4*)(Y + yindex16) = y;
	}
}

#endif


//lengthv % 8 == 0
#ifndef IMG_LINEAR2_DIV_2D_FIELD_KERNEL8
#define IMG_LINEAR2_DIV_2D_FIELD_KERNEL8

//for each field[i]: 
//	Y[i] = (alpha1*X[i] + beta1*X1 + gamma1) / (alpha2*X2 + beta2) + C
//STEP:
//<1> A = beta1*X1 + gamma1
//<2> B = 1 / (alpha2*X2 + beta2)
//<3> K = alpha1*B
//<4> L = A * B + C
//<5> Y = K * X + L

//(5, 3): Size = 40.0156, Time = 0.109 mesc, Speed = 358.512 GB/s
__global__ void img_linear2_div2D_field_kernel_8(
	const char*  __restrict__ X,
	const float* __restrict__ X1,
	const float* __restrict__ X2, int row_lengthv,
	float alpha1, float beta1, float gamma1,
	float alpha2, float beta2, float C,
	float* __restrict__ Y,
	int lengthv, int width, int stride)
{
	int step = gridDim.x*blockDim.x, step8 = step << 3;
	int index = blockIdx.x*blockDim.x + threadIdx.x;

	float4 table[2]; table[0] = F32_0_4;
	for (int index8 = index << 3; index8 < lengthv; index8 += step8)
	{
		uchar8 x = *(uchar8*)(X + index8);

		//when row_lengthv % 8 == 0: field_index{ i = [0, 1] } are the same
		//index16 % 8 == 0, so: index8 = 8x
		//field_index[i] = (index8 + 4*i) / 8 = (8x + 4*i) / 8 = x
		int row_index = index8 / row_lengthv;
		float A = beta1 * X1[row_index] + gamma1;//<1> A = beta1*X1 + gamma1
		float B = 1.0f / (alpha2 * X2[row_index] + beta2);//<2> B = 1 / (alpha2*X2 + beta2)
		float K = alpha1 * B;//<3> K = alpha1*B
		float L = A * B + C;//<4> L = A * B + C
		
		float4 y; int yindex8 = index8;//<5> Y = K * X + L
		y.x = K * x.x0 + L;
		y.y = K * x.y0 + L;
		y.z = K * x.z0 + L;
		y.w = K * x.w0 + L;
		within_width4_zero_nan(y, yindex8, table, stride, width);
		*(float4*)(Y + yindex8) = y; yindex8 += 4;

		y.x = K * x.x1 + L;
		y.y = K * x.y1 + L;
		y.z = K * x.z1 + L;
		y.w = K * x.w1 + L;
		within_width4_zero_nan(y, yindex8, table, stride, width);
		*(float4*)(Y + yindex8) = y;
	}
}

#endif


//common
#ifndef IMG_LINEAR2_DIV_2D_FIELD_KERNEL4
#define IMG_LINEAR2_DIV_2D_FIELD_KERNEL4

//for each field[i]: 
//	Y[i] = (alpha1*X[i] + beta1*X1 + gamma1) / (alpha2*X2 + beta2) + C
//STEP:
//<1> A = beta1*X1 + gamma1
//<2> B = 1 / (alpha2*X2 + beta2)
//<3> K = alpha1*B
//<4> L = A * B + C
//<5> Y = K * X + L

//(5, 3): Size = 40.0156, Time = 0.119 mesc, Speed = 328.385 GB/s
//(5, 2): Size = 40.0156, Time = 0.132 mesc, Speed = 296.044 GB/s
__global__ void img_linear2_div2D_field_kernel_4(
	const char*  __restrict__ X,
	const float* __restrict__ X1,
	const float* __restrict__ X2, int row_lengthv,
	float alpha1, float beta1, float gamma1,
	float alpha2, float beta2, float C,
	float* __restrict__ Y,
	int lengthv, int width, int stride)
{
	int step = gridDim.x*blockDim.x, step4 = step << 2;
	int index = blockIdx.x*blockDim.x + threadIdx.x;

	float4 table[2]; table[0] = F32_0_4;
	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		uchar4 x = *(uchar4*)(X + index4);

		int row_index = index4 / row_lengthv;
		float A = beta1 * X1[row_index] + gamma1;//<1> A = beta1*X1 + gamma1
		float B = 1.0f / (alpha2 * X2[row_index] + beta2);//<2> B = 1 / (alpha2*X2 + beta2)
		float K = alpha1 * B;//<3> K = alpha1*B
		float L = A * B + C;//<4> L = A * B + C

		float4 y;//<5> Y = K * X + L
		y.x = K * x.x + L;
		y.y = K * x.y + L;
		y.z = K * x.z + L;
		y.w = K * x.w + L;

		within_width4_zero_nan(y, index4, table, stride, width);
		*(float4*)(Y + index4) = y;
	}
}

#endif


void __img_linear2_div2D_field(cudaStream_t stream,
	const char*  X,
	const float* X1, 
	const float* X2, int row_lengthv,
	float alpha1, float beta1, float gamma1,
	float alpha2, float beta2, float C,
	float* Y,
	int lengthv, int width, int stride)
{
	if (lengthv < 256) { img_linear2_div2d_field_k4_small(stream, X, X1, X2, row_lengthv, Y, lengthv, width, stride); return; }
	if (!(row_lengthv & 15) && lengthv >= 16384) { img_linear2_div2d_field_k16(stream, 5, 4, X, X1, X2, row_lengthv, Y, lengthv, width, stride); return; }
	if (!(row_lengthv &  7) && lengthv >=  8192) { img_linear2_div2d_field_k8(stream, 5, 3, X, X1, X2, row_lengthv, Y, lengthv, width, stride); return; }
	if (lengthv >= 8192) { img_linear2_div2d_field_k4(stream, 5, 3, X, X1, X2, row_lengthv, Y, lengthv, width, stride); return; }
	img_linear2_div2d_field_k4(stream, 5, 2, X, X1, X2, row_lengthv, Y, lengthv, width, stride);
}

#endif