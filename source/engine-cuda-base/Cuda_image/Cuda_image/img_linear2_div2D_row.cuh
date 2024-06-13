#pragma once

#ifndef IMG_DIV_2D_ROW_H
#define IMG_DIV_2D_ROW_H

//lengthv = height * stride
//stride = (width + 3)/4*4
//[lengthv, row_lengthv] % stride == 0
//field_length * row_lengthv = X1.lengthv
#ifndef IMG_DIV_2D_ROW_CALL
#define IMG_DIV_2D_ROW_CALL

//LB = log2(BLOCK_SIZE)
//LT = log2(THREAD_ELEMENT_VISIT_SIZE)
//LB = log2(BLOCK_SIZE)
//LT = log2(THREAD_ELEMENT_VISIT_SIZE)

#define img_linear2_div2d_row_k4(stream, LB, LT, X, X1, X2, row_lengthv, Y, lengthv, width, stride)\
	img_linear2_div2D_row_kernel_4\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(X, X1, X2, row_lengthv, alpha1, beta1, gamma1, alpha2, beta2, C, Y, lengthv, width, stride)

#define img_linear2_div2d_row_k4_small(stream, X, X1, X2, row_lengthv, Y, lengthv, width, stride)\
	img_linear2_div2D_row_kernel_4\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(X, X1, X2, row_lengthv, alpha1, beta1, gamma1, alpha2, beta2, C, Y, lengthv, width, stride)

#endif


#ifndef IMG_DIV_2D_ROW_KERNEL4
#define IMG_DIV_2D_ROW_KERNEL4

//for each row[i]: 
//	Y[i] = (alpha1*X[i] + beta1*X1 + gamma1) / (alpha2*X2 + beta2) + C
//STEP:
//<1> Y1 = alpha1*X + beta1*X1 + gamma1
//<2> Y2 = alpha2*X2 + beta2
//<3> Y = Y1 / Y2 + C
//
//(5, 3): Size = 40.0156, Time = 0.109 mesc, Speed = 358.512 GB/s
//(5, 2): Size = 40.0156, Time = 0.117 mesc, Speed = 333.998 GB/s
__global__ void img_linear2_div2D_row_kernel_4(
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

		int field_index = index4 % row_lengthv;//X2_lengthv % 4 == 0
		float4 x1 = *(float4*)(X1 + field_index);
		float4 x2 = *(float4*)(X2 + field_index);

		float4 y1;//<1> Y1 = alpha1*X + beta1*X1 + gamma1
		y1.x = alpha1 * x.x + beta1 * x1.x + gamma1;
		y1.y = alpha1 * x.y + beta1 * x1.y + gamma1;
		y1.z = alpha1 * x.z + beta1 * x1.z + gamma1;
		y1.w = alpha1 * x.w + beta1 * x1.w + gamma1;

		float4 y2;//<2> Y2 = alpha2*X2 + beta2
		y2.x = alpha2 * x2.x + beta2;
		y2.y = alpha2 * x2.y + beta2;
		y2.z = alpha2 * x2.z + beta2;
		y2.w = alpha2 * x2.w + beta2;

		float4 y;//Y = Y1 / Y2 + C
		y.x = __fdividef(y1.x, y2.x) + C;
		y.y = __fdividef(y1.y, y2.y) + C;
		y.z = __fdividef(y1.z, y2.z) + C;
		y.w = __fdividef(y1.w, y2.w) + C;
		
		within_width4_zero_nan(y, index4, table, stride, width);
		*(float4*)(Y + index4) = y;
	}
}

#endif


void __img_linear2_div2D_row(cudaStream_t stream,
	const char*  X,
	const float* X1,
	const float* X2, int row_lengthv,
	float alpha1, float beta1, float gamma1,
	float alpha2, float beta2, float C,
	float* Y,
	int lengthv, int width, int stride)
{
	if (lengthv < 256) { img_linear2_div2d_row_k4_small(stream, X, X1, X2, row_lengthv, Y, lengthv, width, stride); return; }
	if (lengthv >= 8192) { img_linear2_div2d_row_k4(stream, 5, 3, X, X1, X2, row_lengthv, Y, lengthv, width, stride); return; }
	img_linear2_div2d_row_k4(stream, 5, 2, X, X1, X2, row_lengthv, Y, lengthv, width, stride);
}

#endif