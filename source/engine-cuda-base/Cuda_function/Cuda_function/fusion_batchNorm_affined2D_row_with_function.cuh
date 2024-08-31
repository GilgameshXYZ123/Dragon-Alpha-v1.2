#pragma once

#ifndef BATCH_NORM_AFFINED_2D_ROW_H_WITH_FUNCTION
#define BATCH_NORM_AFFINED_2D_ROW_H_WITH_FUNCTION

//(1) lengthv = height * stride
//(2) stride = (width + 3)/4*4
//(3) [lenghtv, stride] % 4 == 0
//(4) affined = true
#ifndef BATCH_NORM_AFFINED_2D_ROW_WITH_FUNCTION_CALL
#define BATCH_NORM_AFFINED_2D_ROW_WITH_FUNCTION_CALL

//LB = log2(BLOCK_SIZE)
//LT = log2(THREAD_ELEMENT_VISIT_SIZE)

//lengthv < 256
#define batchNorm_affined2d_row_with_function_k4_small(stream, X, X_mean, X_var, eps, A, B, row_lengthv, Y, lengthv) \
	batchNorm_affined2D_row_with_function_kernel_4<fp32_func_type>\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(X, X_mean, X_var, eps, A, B, row_lengthv, Y, lengthv, width, stride,\
				fp32_func_param0, fp32_func_param1, fp32_func_param2)

//common
#define batchNorm_affined2d_row_with_function_k4(stream, LB, LT, X, X_mean, X_var, eps, A, B, row_lengthv, Y, lengthv) \
	batchNorm_affined2D_row_with_function_kernel_4<fp32_func_type>\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(X, X_mean, X_var, eps, A, B, row_lengthv, Y, lengthv, width, stride,\
				fp32_func_param0, fp32_func_param1, fp32_func_param2)

//lengthv > lengthv_max
#define batchNorm_affined2d_row_with_function_k4_max(stream, X, X_mean, X_var, eps, A, B, row_lengthv, Y, lengthv) \
	batchNorm_affined2D_row_with_function_kernel_4<fp32_func_type>\
		<<< GRID_MAX, 32, 0, stream >>>\
			(X, X_mean, X_var, eps, A, B, row_lengthv, Y, lengthv, width, stride,\
				fp32_func_param0, fp32_func_param1, fp32_func_param2)

#endif


#ifndef BATCH_NORM_AFFINED_2D_ROW_WITH_FUNCTION_KERNEL
#define BATCH_NORM_AFFINED_2D_ROW_WITH_FUNCTION_KERNEL

//=======[Document]==================================================
//<1> N = batch_size =  field_length = lengthv / row_lengthv
//<2> M = row_lengthv
//<3> dims: Y[N, M], X[N, M], X_mean[M], X_var[M], A[M], B[M]
//<4> k: negative_slope
//
//[Forward Propagation]:
//(1) X_mean = mean_each_field(X)
//(2) X_var = variance_each_field(X^2)
//(3) X_std = sqrt(X_var + eps)
//(4) X_norm = (X - X_mean) / X_std
//(5) Y1 = A * X_norm + B
//(6) Y2 = function(Y1)
//=======[Document]==================================================

template<int fp32_func_type>
__global__ void batchNorm_affined2D_row_with_function_kernel_4(
	const float* __restrict__ X,
	const float* __restrict__ X_mean,
	const float* __restrict__ X_var, float eps,
	const float* __restrict__ A,
	const float* __restrict__ B, int row_lengthv,
	      float* __restrict__ Y, 
	int lengthv, int width, int stride,
	float fp32_func_param0, float fp32_func_param1, float fp32_func_param2)
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
		float4 x_norm;//X_norm = (X - X_mean) / sqrt(X_var + eps)
		x_norm.x = (x.x - x_mean.x) * rsqrtf(x_var.x + eps);
		x_norm.y = (x.y - x_mean.y) * rsqrtf(x_var.y + eps);
		x_norm.z = (x.z - x_mean.z) * rsqrtf(x_var.z + eps);
		x_norm.w = (x.w - x_mean.w) * rsqrtf(x_var.w + eps);

		float4 y;//Y = function(X_norm, k)
		y.x = a.x * x_norm.x + b.x; 
		y.y = a.y * x_norm.y + b.y; 
		y.z = a.z * x_norm.z + b.z;
		y.w = a.w * x_norm.w + b.w;

		y.x = fp32_func_forward<fp32_func_type>(y.x, fp32_func_param0, fp32_func_param1, fp32_func_param2);
		y.y = fp32_func_forward<fp32_func_type>(y.y, fp32_func_param0, fp32_func_param1, fp32_func_param2);
		y.z = fp32_func_forward<fp32_func_type>(y.z, fp32_func_param0, fp32_func_param1, fp32_func_param2);
		y.w = fp32_func_forward<fp32_func_type>(y.w, fp32_func_param0, fp32_func_param1, fp32_func_param2);

		//write data-----------------------------------------------------
		within_width_zero_nan(y, index4, table, stride, width);
		*(float4*)(Y + index4) = y;
	}
}

#endif


#ifndef BATCH_NORM_AFFINED_2D_ROW_WITH_FUNCTION
#define BATCH_NORM_AFFINED_2D_ROW_WITH_FUNCTION

template<int fp32_func_type>
void __temp_batchNorm_affined2D_row_with_function(cudaStream_t stream,
	const float* X,
	const float* X_mean,
	const float* X_var, float eps,
	const float* A,
	const float* B, int row_lengthv,
	      float* Y,
	int lengthv, int width, int stride,
	float fp32_func_param0, float fp32_func_param1, float fp32_func_param2)
{
	if (lengthv < 256) { batchNorm_affined2d_row_with_function_k4_small(stream, X, X_mean, X_var, eps, A, B, row_lengthv, Y, lengthv); return; }
	if (lengthv > LENGTHV_MAX) { batchNorm_affined2d_row_with_function_k4_max(stream, X, X_mean, X_var, eps, A, B, row_lengthv, Y, lengthv); return; }
	batchNorm_affined2d_row_with_function_k4(stream, 5, 2, X, X_mean, X_var, eps, A, B, row_lengthv, Y, lengthv);
}


void __batchNorm_affined2D_row_with_function(JNIEnv* env, cudaStream_t stream,
	const float* X,
	const float* X_mean,
	const float* X_var, float eps,
	const float* A,
	const float* B, int row_lengthv,
	       float* Y,
	int lengthv, int width, int stride,
	int fp32_func_type, float fp32_func_param0, float fp32_func_param1, float fp32_func_param2)
{
	if      (fp32_func_type == FP32_Func_Relu)      __temp_batchNorm_affined2D_row_with_function<FP32_Func_Relu>     (stream, X, X_mean, X_var, eps, A, B, row_lengthv, Y, lengthv, width, stride, fp32_func_param0, fp32_func_param1, fp32_func_param2);
	else if (fp32_func_type == FP32_Func_LeakyRelu) __temp_batchNorm_affined2D_row_with_function<FP32_Func_LeakyRelu>(stream, X, X_mean, X_var, eps, A, B, row_lengthv, Y, lengthv, width, stride, fp32_func_param0, fp32_func_param1, fp32_func_param2);
	else if (fp32_func_type == FP32_Func_Elu)       __temp_batchNorm_affined2D_row_with_function<FP32_Func_Elu>      (stream, X, X_mean, X_var, eps, A, B, row_lengthv, Y, lengthv, width, stride, fp32_func_param0, fp32_func_param1, fp32_func_param2);
	else if (fp32_func_type == FP32_Func_Softplus)  __temp_batchNorm_affined2D_row_with_function<FP32_Func_Softplus> (stream, X, X_mean, X_var, eps, A, B, row_lengthv, Y, lengthv, width, stride, fp32_func_param0, fp32_func_param1, fp32_func_param2);
	else if (fp32_func_type == FP32_Func_Gelu)      __temp_batchNorm_affined2D_row_with_function<FP32_Func_Gelu>     (stream, X, X_mean, X_var, eps, A, B, row_lengthv, Y, lengthv, width, stride, fp32_func_param0, fp32_func_param1, fp32_func_param2);
	else if (fp32_func_type == FP32_Func_Sigmoid)   __temp_batchNorm_affined2D_row_with_function<FP32_Func_Sigmoid>  (stream, X, X_mean, X_var, eps, A, B, row_lengthv, Y, lengthv, width, stride, fp32_func_param0, fp32_func_param1, fp32_func_param2);
	else if (fp32_func_type == FP32_Func_Tanh)      __temp_batchNorm_affined2D_row_with_function<FP32_Func_Tanh>     (stream, X, X_mean, X_var, eps, A, B, row_lengthv, Y, lengthv, width, stride, fp32_func_param0, fp32_func_param1, fp32_func_param2);
	else throwException(env, "Unknown FP32_Func_Type");
}

#endif

#endif