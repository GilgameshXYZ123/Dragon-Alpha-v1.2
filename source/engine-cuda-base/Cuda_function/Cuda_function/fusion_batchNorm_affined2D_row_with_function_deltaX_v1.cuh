#pragma once

#ifndef BATCH_NORM_AFFINED_2D_ROW_WITH_FUNCTION_DELTAX_V1_H
#define BATCH_NORM_AFFINED_2D_ROW_WITH_FUNCTION_DELTAX_V1_H

//(1) lengthv = height * stride
//(2) stride = (width + 3)/4*4
//(3) [lenghtv, stride] % 4 == 0
//(4) V1: holdY(), Y is not changed
//(5) affined = true
#ifndef BATCH_NORM_AFFINED_2D_ROW_WITH_FUNCTION_DELTAX_V1_CALL
#define BATCH_NORM_AFFINED_2D_ROW_WITH_FUNCTION_DELTAX_V1_CALL

//LB = log2(BLOCK_SIZE)
//LT = log2(THREAD_ELEMENT_VISIT_SIZE)

//lengthv < 256
#define batchNorm_affined2d_row_with_function_deltaX_v1_k4_small(stream, deltaY, Y, X_var, eps, A, B, deltaXp1, deltaXp2, row_lengthv, deltaX, lengthv)\
	batchNorm_affined2D_row_with_function_deltaX_v1_kernel_4<fp32_func_type>\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(deltaY, Y, X_var, eps, A, B, deltaXp1, deltaXp2, row_lengthv, deltaX, lengthv, width, stride,\
				fp32_func_param0, fp32_func_param1, fp32_func_param2)

//common
#define batchNorm_affined2d_row_with_function_deltaX_v1_k4(stream, LB, LT, deltaY, Y, X_var, eps, A, B, deltaXp1, deltaXp2, row_lengthv, deltaX, lengthv)\
	batchNorm_affined2D_row_with_function_deltaX_v1_kernel_4<fp32_func_type>\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(deltaY, Y, X_var, eps, A, B, deltaXp1, deltaXp2, row_lengthv, deltaX, lengthv, width, stride,\
				fp32_func_param0, fp32_func_param1, fp32_func_param2)

//lengthv > lengthv_max
#define batchNorm_affined2d_row_with_function_deltaX_v1_k4_max(stream, deltaY, Y, X_var, eps, A, B, deltaXp1, deltaXp2, row_lengthv, deltaX, lengthv)\
	batchNorm_affined2D_row_with_function_deltaX_v1_kernel_4<fp32_func_type>\
		<<< GRID_MAX, 32, 0, stream >>>\
			(deltaY, Y, X_var, eps, A, B, deltaXp1, deltaXp2, row_lengthv, deltaX, lengthv, width, stride,\
				fp32_func_param0, fp32_func_param1, fp32_func_param2)

#endif


#ifndef BATCH_NORM_AFFINED_2D_ROW_WITH_FUNCTION_DELTAX_V1_KERNEL
#define BATCH_NORM_AFFINED_2D_ROW_WITH_FUNCTION_DELTAX_V1_KERNEL

//=======[Document]========================================================
//<1> N = batch_size =  field_length = lengthv / row_lengthv
//<2> M = row_lengthv
//<3> dims: Y[N, M], X[N, M], X_mean[M], X_var[M], A[M], B[M]
//<4> dims: deltaY[N, M], deltaX[N, M], deltaXp1[M], deltaXp2[M]
//
//[Forward Propagation]:
//(1) X_mean = mean_each_field(X)
//(2) X_var = variance_each_field(X)
//(3) X_std = sqrt(X_var + eps)
//(4) X_norm = (X - X_mean) / X_std
//(5) Y1 = A * X_norm + B
//(6) Y = Y2 = function(Y1)
//
//[Backward Propagation]
//(1) Y1 = function^(-1)(Y2)
//(2) deltaY1 = deltaY2 * function'(Y1), deltaY2 = deltaY
//(3) X_norm = (Y1 - B) / A
//(4) (deltaXp1 = deltaB) = sum_each_field: deltaY1
//(5) (deltaXp2 = deltaA) = sum_each_field: deltaY1 * Xnorm
//(6) X_rstd = 1 / sqrtf(X_var + eps)
//(7) deltaX = (A * X_rstd) * (deltaY1 - deltaXp1 / N - deltaXp2 * X_norm / N)
//    deltaX = (A * X_rstd) * (deltaY1 - (deltaXp1 + deltaXp2 * X_norm) / N)
//let: rN = (1.0f / N)
//we have: deltaX = (A*X_rstd) * (deltaY1 - rN*(deltaXp1 + deltaXp2*X_norm))
//=======[Document]==========================================================

template<int fp32_func_type>
__global__ void batchNorm_affined2D_row_with_function_deltaX_v1_kernel_4(
	const float* __restrict__ deltaY, 
	const float* __restrict__ Y,
	const float* __restrict__ X_var, float eps,
	const float* __restrict__ A,
	const float* __restrict__ B,
	const float* __restrict__ deltaXp1,
	const float* __restrict__ deltaXp2, int row_lengthv,
	      float* __restrict__ deltaX,
	int lengthv, int width, int stride,
	float fp32_func_param0, float fp32_func_param1, float fp32_func_param2)
{
	const int step = gridDim.x*blockDim.x, step4 = step << 2;
	const int index = blockIdx.x*blockDim.x + threadIdx.x;

	const float rN = (1.0f * row_lengthv) / lengthv;//rN = (1.0f / N)
	float4 table[2]; table[0] = F32_4_0;//(A == 0) will cause NaN
	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		//load data------------------------------------------------------------
		const int field_index4 = index4 % row_lengthv;
		const float4 dy2 = *(float4*)(deltaY + index4);
		const float4 y2 = *(float4*)(Y + index4);
		const float4 x_var = *(float4*)(X_var + field_index4);
		const float4 a = *(float4*)(A + field_index4);
		const float4 b = *(float4*)(B + field_index4);
		const float4 dxp1 = *(float4*)(deltaXp1 + field_index4);
		const float4 dxp2 = *(float4*)(deltaXp2 + field_index4);

		//compute result-------------------------------------------------------
		float4 y1;//Y1 = function^(-1)(Y2)
		y1.x = fp32_func_inverse<fp32_func_type>(y2.x, fp32_func_param0, fp32_func_param1, fp32_func_param2);
		y1.y = fp32_func_inverse<fp32_func_type>(y2.y, fp32_func_param0, fp32_func_param1, fp32_func_param2);
		y1.z = fp32_func_inverse<fp32_func_type>(y2.z, fp32_func_param0, fp32_func_param1, fp32_func_param2);
		y1.w = fp32_func_inverse<fp32_func_type>(y2.w, fp32_func_param0, fp32_func_param1, fp32_func_param2);

		float4 dy1;//deltaY1 = deltaY2 * function'(Y1)
		dy1.x = dy2.x * fp32_func_derivative_v2<fp32_func_type>(y1.x, fp32_func_param0, fp32_func_param1, fp32_func_param2);
		dy1.y = dy2.y * fp32_func_derivative_v2<fp32_func_type>(y1.y, fp32_func_param0, fp32_func_param1, fp32_func_param2);
		dy1.z = dy2.z * fp32_func_derivative_v2<fp32_func_type>(y1.z, fp32_func_param0, fp32_func_param1, fp32_func_param2);
		dy1.w = dy2.w * fp32_func_derivative_v2<fp32_func_type>(y1.w, fp32_func_param0, fp32_func_param1, fp32_func_param2);

		float4 x_norm;//X_norm = (Y - B) / A
		x_norm.x = (y1.x - b.x) / a.x;
		x_norm.y = (y1.y - b.y) / a.y;
		x_norm.z = (y1.z - b.z) / a.z;
		x_norm.w = (y1.w - b.w) / a.w;

		float4 dx;//deltaX = (A*X_rstd) * (deltaY - rN*(deltaXp1 + deltaXp2*X_norm))
		dx.x = (a.x * rsqrtf(x_var.x + eps)) * (dy1.x - rN * (dxp1.x + dxp2.x * x_norm.x));
		dx.y = (a.y * rsqrtf(x_var.y + eps)) * (dy1.y - rN * (dxp1.y + dxp2.y * x_norm.y));
		dx.z = (a.z * rsqrtf(x_var.z + eps)) * (dy1.z - rN * (dxp1.z + dxp2.z * x_norm.z));
		dx.w = (a.w * rsqrtf(x_var.w + eps)) * (dy1.w - rN * (dxp1.w + dxp2.w * x_norm.w));

		//write data-----------------------------------------------------------
		within_width_zero_nan(dx, index4, table, stride, width);
		*(float4*)(deltaX + index4) = dx;
	}
}

#endif


#ifndef BATCH_NORM_AFFINED_2D_ROW_WITH_FUNCTION_DELTAX_V1
#define BATCH_NORM_AFFINED_2D_ROW_WITH_FUNCTION_DELTAX_V1

template<int fp32_func_type>
void __temp_batchNorm_affined2D_row_with_function_deltaX_v1(cudaStream_t stream,
	const float* deltaY, 
	const float* Y,
	const float* X_var, float eps,
	const float* A,
	const float* B,
	const float* deltaXp1,
	const float* deltaXp2, int row_lengthv,
	      float* deltaX,
	int lengthv, int width, int stride,
	float fp32_func_param0, float fp32_func_param1, float fp32_func_param2)
{
	if (lengthv < 256) { batchNorm_affined2d_row_with_function_deltaX_v1_k4_small(stream, deltaY, Y, X_var, eps, A, B, deltaXp1, deltaXp2, row_lengthv, deltaX, lengthv); return; }
	if (lengthv > LENGTHV_MAX) { batchNorm_affined2d_row_with_function_deltaX_v1_k4_max(stream, deltaY, Y, X_var, eps, A, B, deltaXp1, deltaXp2, row_lengthv, deltaX, lengthv); return; }
	batchNorm_affined2d_row_with_function_deltaX_v1_k4(stream, 5, 2, deltaY, Y, X_var, eps, A, B, deltaXp1, deltaXp2, row_lengthv, deltaX, lengthv);
}


void __batchNorm_affined2D_row_with_function_deltaX_v1(JNIEnv* env, cudaStream_t stream,
	const float* deltaY,
	const float* Y,
	const float* X_var, float eps,
	const float* A,
	const float* B,
	const float* deltaXp1,
	const float* deltaXp2, int row_lengthv,
	      float* deltaX,
	int lengthv, int width, int stride,
	int fp32_func_type, float fp32_func_param0, float fp32_func_param1, float fp32_func_param2)
{
	if      (fp32_func_type == FP32_Func_Relu)      __temp_batchNorm_affined2D_row_with_function_deltaX_v1<FP32_Func_Relu>     (stream, deltaY, Y, X_var, eps, A, B, deltaXp1, deltaXp2, row_lengthv, deltaX, lengthv, width, stride, fp32_func_param0, fp32_func_param1, fp32_func_param2);
	else if (fp32_func_type == FP32_Func_LeakyRelu) __temp_batchNorm_affined2D_row_with_function_deltaX_v1<FP32_Func_LeakyRelu>(stream, deltaY, Y, X_var, eps, A, B, deltaXp1, deltaXp2, row_lengthv, deltaX, lengthv, width, stride, fp32_func_param0, fp32_func_param1, fp32_func_param2);
	else if (fp32_func_type == FP32_Func_Elu)       __temp_batchNorm_affined2D_row_with_function_deltaX_v1<FP32_Func_Elu>      (stream, deltaY, Y, X_var, eps, A, B, deltaXp1, deltaXp2, row_lengthv, deltaX, lengthv, width, stride, fp32_func_param0, fp32_func_param1, fp32_func_param2);
	else if (fp32_func_type == FP32_Func_Softplus)  __temp_batchNorm_affined2D_row_with_function_deltaX_v1<FP32_Func_Softplus> (stream, deltaY, Y, X_var, eps, A, B, deltaXp1, deltaXp2, row_lengthv, deltaX, lengthv, width, stride, fp32_func_param0, fp32_func_param1, fp32_func_param2);
	else if (fp32_func_type == FP32_Func_Gelu)      __temp_batchNorm_affined2D_row_with_function_deltaX_v1<FP32_Func_Gelu>     (stream, deltaY, Y, X_var, eps, A, B, deltaXp1, deltaXp2, row_lengthv, deltaX, lengthv, width, stride, fp32_func_param0, fp32_func_param1, fp32_func_param2);
	else if (fp32_func_type == FP32_Func_Sigmoid)   __temp_batchNorm_affined2D_row_with_function_deltaX_v1<FP32_Func_Sigmoid>  (stream, deltaY, Y, X_var, eps, A, B, deltaXp1, deltaXp2, row_lengthv, deltaX, lengthv, width, stride, fp32_func_param0, fp32_func_param1, fp32_func_param2);
	else if (fp32_func_type == FP32_Func_Tanh)      __temp_batchNorm_affined2D_row_with_function_deltaX_v1<FP32_Func_Tanh>     (stream, deltaY, Y, X_var, eps, A, B, deltaXp1, deltaXp2, row_lengthv, deltaX, lengthv, width, stride, fp32_func_param0, fp32_func_param1, fp32_func_param2);
	else throwException(env, "Unknown FP32_Func_Type");
}

#endif

#endif