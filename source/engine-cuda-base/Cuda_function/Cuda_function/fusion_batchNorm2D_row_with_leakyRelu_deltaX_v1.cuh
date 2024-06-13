#pragma once

#ifndef BATCH_NORM_2D_ROW_WITH_LEAKY_RELU_DELTAX_V1_H
#define BATCH_NORM_2D_ROW_WITH_LEAKY_RELU_DELTAX_V1_H

//(1) lengthv = height * stride
//(2) stride = (width + 3)/4*4
//(3) [lenghtv, stride] % 4 == 0
//(4) V1: holdY(), Y is not changed
//(5) affined = false
#ifndef BATCH_NORM_2D_ROW_WITH_LEAKY_RELU_DELTAX_V1_CALL
#define BATCH_NORM_2D_ROW_WITH_LEAKY_RELU_DELTAX_V1_CALL

//LB = log2(BLOCK_SIZE)
//LT = log2(THREAD_ELEMENT_VISIT_SIZE)

//lengthv < 256
#define batchNorm2d_row_with_leakyRelu_deltaX_v1_k4_small(stream, deltaY, Y, k, X_var, eps, deltaXp1, deltaXp2, row_lengthv, deltaX, lengthv)\
	batchNorm2D_row_with_leakyRelu_deltaX_v1_kernel_4\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(deltaY, Y, k, X_var, eps, deltaXp1, deltaXp2, row_lengthv, deltaX, lengthv, width, stride)

//common
#define batchNorm2d_row_with_leakyRelu_deltaX_v1_k4(stream, LB, LT, deltaY, Y, k, X_var, eps, deltaXp1, deltaXp2, row_lengthv, deltaX, lengthv)\
	batchNorm2D_row_with_leakyRelu_deltaX_v1_kernel_4\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(deltaY, Y, k, X_var, eps, deltaXp1, deltaXp2, row_lengthv, deltaX, lengthv, width, stride)

//lengthv > lengthv_max
#define batchNorm2d_row_with_leakyRelu_deltaX_v1_k4_max(stream, deltaY, Y, k, X_var, eps, deltaXp1, deltaXp2, row_lengthv, deltaX, lengthv)\
	batchNorm2D_row_with_leakyRelu_deltaX_v1_kernel_4\
		<<< GRID_MAX, 32, 0, stream >>>\
			(deltaY, Y, k, X_var, eps, deltaXp1, deltaXp2, row_lengthv, deltaX, lengthv, width, stride)

#endif


#ifndef BATCH_NORM_2D_ROW_WITH_LEAKY_RELU_DELTAX_V1_KERNEL
#define BATCH_NORM_2D_ROW_WITH_LEAKY_RELU_DELTAX_V1_KERNEL

//=======[Document]==================================================
//<1> N = batch_size =  field_length = lengthv / row_lengthv
//<2> M = row_lengthv
//<3> dims: Y[N, M], X[N, M], X_mean[M], X_var[M], A[M], B[M]
//<4> dims: deltaY[N, M], deltaX[N, M], deltaXp1[M], deltaXp2[M]
//
//[Forward Propagation]:
//(1) X_mean = mean_each_field(X)
//(2) X_var = variance_each_field(X)
//(3) X_std = sqrt(X_var + eps)
//(4) Y1 = X_norm = (X - X_mean) / X_std
//(6) Y2 = leakyRelu(Y1, k)
//
//[Backward Propagation]
//(1) X_norm = Y1 = Y2 * (Y2 > 0 ? 1 : rk)
//(2) deltaY1 = deltaY2 * (Y2 > 0 ? 1 :  k), deltaY2 = deltaY
//(3) deltaXp1 = sum_each_field: deltaY1
//(4) deltaXp2 = sum_each_field: deltaY1 * Y1
//(5) X_rstd = 1 / sqrtf(X_var + eps)
//(6) deltaX = X_rstd * (deltaY1 - deltaXp1 / N - deltaXp2 * X_norm / N)
//    deltaX = X_rstd * (deltaY1 - (deltaXp1 + deltaXp2*X_norm) / N)
//let: rN = (1.0f / N)
//we have: deltaX = X_rstd * (deltaY - rN*(deltaXp1 + deltaXp2*X_norm))
//=======[Document]==================================================

__global__ void batchNorm2D_row_with_leakyRelu_deltaX_v1_kernel_4(
	const float* __restrict__ deltaY,
	const float* __restrict__ Y, float k,
	const float* __restrict__ X_var, float eps,
	const float* __restrict__ deltaXp1,
	const float* __restrict__ deltaXp2, int row_lengthv,
	      float* __restrict__ deltaX,
	int lengthv, int width, int stride)
{
	const int step = gridDim.x*blockDim.x, step4 = step << 2;
	const int index = blockIdx.x*blockDim.x + threadIdx.x;

	const float rK = 1.0f / k;
	const float rN = (1.0f * row_lengthv) / lengthv;//(1) rN = (1.0f / N)
	float4 table[2]; table[0] = F32_4_0;//(A == 0) will cause NaN
	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		//load data------------------------------------------------------------
		const int field_index4 = index4 % row_lengthv;
		const float4 dy2 = *(float4*)(deltaY + index4);
		const float4 y2 = *(float4*)(Y + index4);
		const float4 x_var = *(float4*)(X_var + field_index4);
		const float4 dxp1 = *(float4*)(deltaXp1 + field_index4);
		const float4 dxp2 = *(float4*)(deltaXp2 + field_index4);

		//compute result-------------------------------------------------------
		char4 flag;//flag = (Y2 > 0);
		flag.x = (y2.x > 0.0f);
		flag.y = (y2.y > 0.0f);
		flag.z = (y2.z > 0.0f);
		flag.w = (y2.w > 0.0f);

		float4 x_norm;//X_norm = Y1 = Y2 * (Y2 > 0 ? 1 : rk)
		x_norm.x = y2.x * (flag.x + !flag.x*rK);
		x_norm.y = y2.y * (flag.y + !flag.y*rK);
		x_norm.z = y2.z * (flag.z + !flag.z*rK);
		x_norm.w = y2.w * (flag.w + !flag.w*rK);

		float4 dy1;//deltaY1 = deltaY2 * (Y2 > 0 ? 1 : k), deltaY2 = deltaY
		dy1.x = dy2.x * (flag.x + !flag.x*k);
		dy1.y = dy2.y * (flag.y + !flag.y*k);
		dy1.z = dy2.z * (flag.z + !flag.z*k);
		dy1.w = dy2.w * (flag.w + !flag.w*k);

		float4 dx;//deltaX = X_rstd * (deltaY - rN*(deltaXp1 + deltaXp2*X_norm))
		dx.x = rsqrtf(x_var.x + eps) * (dy1.x - rN * (dxp1.x + dxp2.x * x_norm.x));
		dx.y = rsqrtf(x_var.y + eps) * (dy1.y - rN * (dxp1.y + dxp2.y * x_norm.y));
		dx.z = rsqrtf(x_var.z + eps) * (dy1.z - rN * (dxp1.z + dxp2.z * x_norm.z));
		dx.w = rsqrtf(x_var.w + eps) * (dy1.w - rN * (dxp1.w + dxp2.w * x_norm.w));

		//write data-----------------------------------------------------------
		within_width_zero_nan(dx, index4, table, stride, width);
		*(float4*)(deltaX + index4) = dx;
	}
}

#endif


void __batchNorm2D_row_with_leakyRelu_deltaX_v1(cudaStream_t stream,
	const float* deltaY,
	const float* Y, float k,
	const float* X_var, float eps,
	const float* deltaXp1,
	const float* deltaXp2, int row_lengthv,
	      float* deltaX,
	int lengthv, int width, int stride)
{
	if (lengthv < 256) { batchNorm2d_row_with_leakyRelu_deltaX_v1_k4_small(stream, deltaY, Y, k, X_var, eps, deltaXp1, deltaXp2, row_lengthv, deltaX, lengthv); return; }
	if (lengthv > LENGTHV_MAX) { batchNorm2d_row_with_leakyRelu_deltaX_v1_k4_max(stream, deltaY, Y, k, X_var, eps, deltaXp1, deltaXp2, row_lengthv, deltaX, lengthv); return; }
	batchNorm2d_row_with_leakyRelu_deltaX_v1_k4(stream, 5, 2, deltaY, Y, k, X_var, eps, deltaXp1, deltaXp2, row_lengthv, deltaX, lengthv);
}

#endif