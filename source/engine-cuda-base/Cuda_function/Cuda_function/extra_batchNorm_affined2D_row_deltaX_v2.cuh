#pragma once

#ifndef BATCH_NORM_AFFINED_2D_ROW_DELTAX_V2_H
#define BATCH_NORM_AFFINED_2D_ROW_DELTAX_V2_H

//(1) lengthv = height * stride
//(2) stride = (width + 3)/4*4
//(3) [lenghtv, stride] % 4 == 0
//(4) V2: holdX(), X is not changed
//(5) affined = true
#ifndef BATCH_NORM_AFFINED_2D_ROW_DELTAX_V2_CALL
#define BATCH_NORM_AFFINED_2D_ROW_DELTAX_V2_CALL

//LB = log2(BLOCK_SIZE)
//LT = log2(THREAD_ELEMENT_VISIT_SIZE)

//lengthv < 256
#define batchnorm_affined2d_row_deltaX_v2_k4_small(stream, deltaY, X, X_mean, X_var, eps, A, deltaXp1, deltaXp2, row_lengthv, deltaX, lengthv, width, stride)\
	batchNorm_affined2D_row_deltaX_v2_kernel_4\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(deltaY, X, X_mean, X_var, eps, A, deltaXp1, deltaXp2, row_lengthv, deltaX, lengthv, width, stride)

//common
#define batchNorm_affined2d_row_deltaX_v2_k4(stream, LB, LT, deltaY, X, X_mean, X_var, eps, A, deltaXp1, deltaXp2, row_lengthv, deltaX, lengthv, width, stride)\
	batchNorm_affined2D_row_deltaX_v2_kernel_4\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(deltaY, X, X_mean, X_var, eps, A, deltaXp1, deltaXp2, row_lengthv, deltaX, lengthv, width, stride)

//lengthv > lengthv_max
#define batchnorm_affined2d_row_deltaX_v2_k4_max(stream, deltaY, X, X_mean, X_var, eps, A, deltaXp1, deltaXp2, row_lengthv, deltaX, lengthv, width, stride)\
	batchNorm_affined2D_row_deltaX_v2_kernel_4\
		<<< GRID_MAX, 32, 0, stream >>>\
			(deltaY, X, X_mean, X_var, eps, A, deltaXp1, deltaXp2, row_lengthv, deltaX, lengthv, width, stride)

#endif


#ifndef BATCH_NORM_AFFINED_2D_ROW_DELTAX_V2_KERNEL
#define BATCH_NORM_AFFINED_2D_ROW_DELTAX_V2_KERNEL

//=======[Document]========================================================
//<1> N = batch_size =  field_length = lengthv / row_lengthv
//<2> M = row_lengthv
//<3> dims: Y[N, M], X[N, M], X_mean[M], X_var[M], A[M], B[M]
//<4> dims: deltaY[N, M], deltaX[N, M], deltaXp1[M], deltaXp2[M]
//
//[Forward Propagation]:
//(1) X_mean = mean_each_field(X)
//(2) X_var = variance_each_field(X)
//(3) X_std = sqrt(X_var - X_mean^2 + eps)
//(4) X_norm = (X - X_mean) / X_std
//(5) Y = A * X_norm + B
//
//[Backward Propagation]
//(1) (deltaXp1 = deltaB) = sum_each_field: deltaY
//(2) (deltaXp2 = deltaA) = sum_each_field: deltaY * Xnorm
//(3) X_rstd = 1 / X_std
//(4) deltaX = (A * X_rstd) * (deltaY - deltaXp1 / N - deltaXp2 * X_norm / N)
//    deltaX = (A * X_rstd) * (deltaY - (deltaXp1 + deltaXp2 * X_norm) / N)
//STEP:
//(1) rN = (1.0f / N)
//(2) X_rstd = rsqrtf(X_var + eps)
//(3) X_norm = (X - X_mean) * X_rstd
//(4) deltaX = (A*X_rstd) * (deltaY - rN*(deltaXp1 + deltaXp2*X_norm))
//=======[Document]========================================================

__global__ void batchNorm_affined2D_row_deltaX_v2_kernel_4(
	const float* __restrict__ deltaY,
	const float* __restrict__ X,
	const float* __restrict__ X_mean,
	const float* __restrict__ X_var, float eps,
	const float* __restrict__ A,
	const float* __restrict__ deltaXp1,
	const float* __restrict__ deltaXp2, int row_lengthv,
	      float* __restrict__ deltaX,
	int lengthv, int width, int stride)
{
	const int step = gridDim.x*blockDim.x, step4 = step << 2;
	const int index = blockIdx.x*blockDim.x + threadIdx.x;

	const float rN = (1.0f * row_lengthv) / lengthv;//(1) rN = (1.0f / N)
	float4 table[2]; table[0] = F32_4_0;//(x_var == 0) will cause NaN
	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		//load data------------------------------------------------------------
		const int field_index4 = index4 % row_lengthv;
		const float4 dy = *(float4*)(deltaY + index4);
		const float4 x = *(float4*)(X + index4);
		const float4 x_mean = *(float4*)(X_mean + field_index4);
		const float4 x_var = *(float4*)(X_var + field_index4);
		const float4 a = *(float4*)(A + field_index4);
		const float4 dxp1 = *(float4*)(deltaXp1 + field_index4);
		const float4 dxp2 = *(float4*)(deltaXp2 + field_index4);

		//compute result-------------------------------------------------------
		float4 x_rstd;//(2) X_rstd = rsqrtf(X_var + eps)
		x_rstd.x = rsqrtf(x_var.x + eps);
		x_rstd.y = rsqrtf(x_var.y + eps);
		x_rstd.z = rsqrtf(x_var.z + eps);
		x_rstd.w = rsqrtf(x_var.w + eps);

		float4 x_norm;//(3) X_norm = (X - X_mean) * X_rstd
		x_norm.x = (x.x - x_mean.x) * x_rstd.x;
		x_norm.y = (x.y - x_mean.y) * x_rstd.y;
		x_norm.z = (x.z - x_mean.z) * x_rstd.z;
		x_norm.w = (x.w - x_mean.w) * x_rstd.w;

		float4 dx;//(4) deltaX = (A*X_rstd) * (deltaY - rN*(deltaXp1 + deltaXp2*X_norm))
		dx.x = (a.x * x_rstd.x) * (dy.x - rN * (dxp1.x + dxp2.x * x_norm.x));
		dx.y = (a.y * x_rstd.y) * (dy.y - rN * (dxp1.y + dxp2.y * x_norm.y));
		dx.z = (a.z * x_rstd.z) * (dy.z - rN * (dxp1.z + dxp2.z * x_norm.z));
		dx.w = (a.w * x_rstd.w) * (dy.w - rN * (dxp1.w + dxp2.w * x_norm.w));

		//write data-----------------------------------------------------------
		within_width_zero_nan(dx, index4, table, stride, width);
		*(float4*)(deltaX + index4) = dx;
	}
}

#endif


void __batchNorm_affined2D_row_deltaX_v2(cudaStream_t stream,
	const float* deltaY, const float* X,
	const float* X_mean,
	const float* X_var, float eps,
	const float* A,
	const float* deltaXp1,
	const float* deltaXp2, int row_lengthv,
	      float* deltaX,
	int lengthv, int width, int stride)
{
	if (lengthv < 256) { batchnorm_affined2d_row_deltaX_v2_k4_small(stream, deltaY, X, X_mean, X_var, eps, A, deltaXp1, deltaXp2, row_lengthv, deltaX, lengthv, width, stride); return; }
	if (lengthv > LENGTHV_MAX) { batchnorm_affined2d_row_deltaX_v2_k4_max(stream, deltaY, X, X_mean, X_var, eps, A, deltaXp1, deltaXp2, row_lengthv, deltaX, lengthv, width, stride); return; }
	batchNorm_affined2d_row_deltaX_v2_k4(stream, 5, 2, deltaY, X, X_mean, X_var, eps, A, deltaXp1, deltaXp2, row_lengthv, deltaX, lengthv, width, stride);
}

#endif