#pragma once

#ifndef BATCH_NORM_AFFINED_2D_ROW_H
#define BATCH_NORM_AFFINED_2D_ROW_H

//(1) lengthv = height * stride
//(2) stride = (width + 3)/4*4
//(3) [lenghtv, stride] % 4 == 0
//(4) affined = true
#ifndef BATCH_NORM_AFFINED_2D_ROW_CALL
#define BATCH_NORM_AFFINED_2D_ROW_CALL

//LB = log2(BLOCK_SIZE)
//LT = log2(THREAD_ELEMENT_VISIT_SIZE)

//lengthv < 256
#define batchNorm_affined2d_row_k4_small(stream, X, X_mean, X_var, eps, A, B, row_lengthv, Y, lengthv, width, stride)\
	batchNorm_affined2D_row_kernel_4\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(X, X_mean, X_var, eps, A, B, row_lengthv, Y, lengthv, width, stride)

//common
#define batchNorm_affined2d_row_k4(stream, LB, LT, X, X_mean, X_var, eps, A, B, row_lengthv, Y, lengthv, width, stride)\
	batchNorm_affined2D_row_kernel_4\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(X, X_mean, X_var, eps, A, B, row_lengthv, Y, lengthv, width, stride)

//lengthv > lengthv_max
#define batchNorm_affined2d_row_k4_max(stream, X, X_mean, X_var, eps, A, B, row_lengthv, Y, lengthv, width, stride)\
	batchNorm_affined2D_row_kernel_4\
		<<< GRID_MAX, 32, 0, stream >>>\
			(X, X_mean, X_var, eps, A, B, row_lengthv, Y, lengthv, width, stride)

#endif


#ifndef BATCH_NORM_AFFINED_2D_ROW_KERNEL
#define BATCH_NORM_AFFINED_2D_ROW_KERNEL

//=======[Document]==================================================
//<1> N = batch_size =  field_length = lengthv / row_lengthv
//<2> M = row_lengthv
//<3> dims: Y[N, M], X[N, M], X_mean[M], X_var[M], A[M], B[M]
//
//[Forward Propagation]:
//(1) X_mean = mean_each_field(X)
//(2) X_var = variance_each_field(X^2)
//(3) X_std = sqrt(X_var + eps)
//(4) X_norm = (X - X_mean) / X_std
//(5) Y = A * X_norm + B
//=======[Document]==================================================

__global__ void batchNorm_affined2D_row_kernel_4(
	const float* __restrict__ X,
	const float* __restrict__ X_mean,
	const float* __restrict__ X_var, float eps,
	const float* __restrict__ A,
	const float* __restrict__ B, int row_lengthv,
	float* __restrict__ Y,
	int lengthv, int width, int stride)
{
	const int step = gridDim.x*blockDim.x, step4 = step << 2;
	const int index = blockIdx.x*blockDim.x + threadIdx.x;

	float4 table[2]; table[0] = F32_4_0;//(x_var == 0) will cause NaN
	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		//load data------------------------------------------------------
		const int field_index4 = index4 % row_lengthv;
		const float4 x = *(float4*)(X + index4);
		const float4 x_mean = *(float4*)(X_mean + field_index4);
		const float4 x_var = *(float4*)(X_var + field_index4);
		const float4 a = *(float4*)(A + field_index4);
		const float4 b = *(float4*)(B + field_index4);

		//compute result-------------------------------------------------
		float4 x_norm;//(1) X_norm = (X - X_mean) / sqrt(X_var + eps)
		x_norm.x = (x.x - x_mean.x) * rsqrtf(x_var.x + eps);
		x_norm.y = (x.y - x_mean.y) * rsqrtf(x_var.y + eps);
		x_norm.z = (x.z - x_mean.z) * rsqrtf(x_var.z + eps);
		x_norm.w = (x.w - x_mean.w) * rsqrtf(x_var.w + eps);

		float4 y;//(2) Y = A * X_norm + B
		y.x = a.x * x_norm.x + b.x;
		y.y = a.y * x_norm.y + b.y;
		y.z = a.z * x_norm.z + b.z;
		y.w = a.w * x_norm.w + b.w;

		//if (y.x != y.x) printf("%f, [%f, %f, %f], %f, %f, %d\n", y.x, x_norm.x, x_mean.x, x_var.x, a.x, b.x, field_index4);
		//if (y.y != y.y) printf("%f, [%f, %f, %f], %f, %f, %d\n", y.y, x_norm.y, x_mean.y, x_var.y, a.y, b.y, field_index4);
		//if (y.z != y.z) printf("%f, [%f, %f, %f], %f, %f, %d\n", y.z, x_norm.z, x_mean.z, x_var.z, a.z, b.z, field_index4);
		//if (y.w != y.w) printf("%f, [%f, %f, %f], %f, %f, %d\n", y.w, x_norm.w, x_mean.w, x_var.w, a.w, b.w, field_index4);

		//write data-----------------------------------------------------
		within_width_zero_nan(y, index4, table, stride, width);
		*(float4*)(Y + index4) = y;
	}
}

#endif


void __batchNorm_affined2D_row(cudaStream_t stream,
	const float* X,
	const float* X_mean,
	const float* X_var, float eps,
	const float* A,
	const float* B, int row_lengthv,
	      float* Y,
	int lengthv, int width, int stride)
{
	if (lengthv < 256) { batchNorm_affined2d_row_k4_small(stream, X, X_mean, X_var, eps, A, B, row_lengthv, Y, lengthv, width, stride); return; }
	if (lengthv > LENGTHV_MAX) { batchNorm_affined2d_row_k4_max(stream, X, X_mean, X_var, eps, A, B, row_lengthv, Y, lengthv, width, stride); return; }
	batchNorm_affined2d_row_k4(stream, 5, 2, X, X_mean, X_var, eps, A, B, row_lengthv, Y, lengthv, width, stride);
}

#endif