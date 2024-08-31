#pragma once

#ifndef AFFINE_2D_ROW_WITH_FUNCTION_H
#define AFFINE_2D_ROW_WITH_FUNCTION_H

//lengthv = height * stride
//stride = (width + 3)/4*4
//[lenghtv, stride] % 4 == 0
#ifndef AFFINE_2D_ROW_WITH_FUNCTION_CALL
#define AFFINE_2D_ROW_WITH_FUNCTION_CALL

//LB = log2(BLOCK_SIZE)
//LT = log2(THREAD_ELEMENT_VISIT_SIZE)

//lengthv < 256
#define affine2d_row_with_function_k4_small(stream, X, A, B, row_lengthv, Y, lengthv, width, stride)\
	affine2D_row_with_function_kernel_4<fp32_func_type>\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(X, A, B, row_lengthv, Y, lengthv, width, stride,\
				fp32_func_param0, fp32_func_param1, fp32_func_param2)

//common
#define affine2d_row_with_function_k4(stream, LB, LT, X, A, B, row_lengthv, Y, lengthv, width, stride)\
	affine2D_row_with_function_kernel_4<fp32_func_type>\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(X, A, B, row_lengthv, Y, lengthv, width, stride,\
				fp32_func_param0, fp32_func_param1, fp32_func_param2)

//lengthv > lengthv_max
#define affine2d_row_with_function_k4_max(stream, X, A, B, row_lengthv, Y, lengthv, width, stride)\
	affine2D_row_with_function_kernel_4<fp32_func_type>\
		<<< GRID_MAX, 32, 0, stream >>>\
			(X, A, B, row_lengthv, Y, lengthv, width, stride,\
				fp32_func_param0, fp32_func_param1, fp32_func_param2)

#endif


#ifndef AFFINE_2D_ROW_WITH_FUNCTION_KERNEL
#define AFFINE_2D_ROW_WITH_FUNCTION_KERNEL

//X2_lengthv = X_mean.lengthv = X_square_mean.lengthv
template<int fp32_func_type>
__global__ void affine2D_row_with_function_kernel_4(
	const float* __restrict__ X,
	const float* __restrict__ A,
	const float* __restrict__ B, int row_lengthv,
	      float* __restrict__ Y,
	int lengthv, int width, int stride,
	float fp32_func_param0, float fp32_func_param1, float fp32_func_param2)
{
	const int step = gridDim.x*blockDim.x, step4 = step << 2;
	const int index = blockIdx.x*blockDim.x + threadIdx.x;

	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		const float4 x = *(float4*)(X + index4);

		const int field_index4 = index4 % row_lengthv;
		const float4 a = *(float4*)(A + field_index4);
		const float4 b = *(float4*)(B + field_index4);

		float4 y;//Y[i] = function(A * X[i] + B)
		y.x = (a.x * x.x) + b.x; 
		y.y = (a.y * x.y) + b.y; 
		y.z = (a.z * x.z) + b.z; 
		y.w = (a.w * x.w) + b.w;

		y.x = fp32_func_forward<fp32_func_type>(y.x, fp32_func_param0, fp32_func_param1, fp32_func_param2);
		y.y = fp32_func_forward<fp32_func_type>(y.y, fp32_func_param0, fp32_func_param1, fp32_func_param2);
		y.z = fp32_func_forward<fp32_func_type>(y.z, fp32_func_param0, fp32_func_param1, fp32_func_param2);
		y.w = fp32_func_forward<fp32_func_type>(y.w, fp32_func_param0, fp32_func_param1, fp32_func_param2);

		within_width(y, index4, stride, width);
		*(float4*)(Y + index4) = y;
	}
}

#endif


#ifndef AFFINE_2D_ROW_WITH_FUNCTION
#define AFFINE_2D_ROW_WITH_FUNCTION

template<int fp32_func_type>
void __temp_affine2D_row_with_function(cudaStream_t stream,
	const float* X,
	const float* A,
	const float* B, int row_lengthv,
	      float* Y,
	int lengthv, int width, int stride,
	float fp32_func_param0, float fp32_func_param1, float fp32_func_param2)
{
	if (lengthv < 256) { affine2d_row_with_function_k4_small(stream, X, A, B, row_lengthv, Y, lengthv, width, stride); return; }
	if (lengthv > LENGTHV_MAX) { affine2d_row_with_function_k4_max(stream, X, A, B, row_lengthv, Y, lengthv, width, stride); return; }
	affine2d_row_with_function_k4(stream, 5, 2, X, A, B, row_lengthv, Y, lengthv, width, stride);
}


void __affine2D_row_with_function(JNIEnv* env, cudaStream_t stream,
	const float* X,
	const float* A,
	const float* B, int row_lengthv,
	      float* Y,
	int lengthv, int width, int stride,
	int fp32_func_type, float fp32_func_param0, float fp32_func_param1, float fp32_func_param2)
{
	if      (fp32_func_type == FP32_Func_Relu)      __temp_affine2D_row_with_function<FP32_Func_Relu>     (stream, X, A, B, row_lengthv, Y, lengthv, width, stride, fp32_func_param0, fp32_func_param1, fp32_func_param2);
	else if (fp32_func_type == FP32_Func_LeakyRelu) __temp_affine2D_row_with_function<FP32_Func_LeakyRelu>(stream, X, A, B, row_lengthv, Y, lengthv, width, stride, fp32_func_param0, fp32_func_param1, fp32_func_param2);
	else if (fp32_func_type == FP32_Func_Elu)       __temp_affine2D_row_with_function<FP32_Func_Elu>      (stream, X, A, B, row_lengthv, Y, lengthv, width, stride, fp32_func_param0, fp32_func_param1, fp32_func_param2);
	else if (fp32_func_type == FP32_Func_Softplus)  __temp_affine2D_row_with_function<FP32_Func_Softplus> (stream, X, A, B, row_lengthv, Y, lengthv, width, stride, fp32_func_param0, fp32_func_param1, fp32_func_param2);
	else if (fp32_func_type == FP32_Func_Gelu)      __temp_affine2D_row_with_function<FP32_Func_Gelu>     (stream, X, A, B, row_lengthv, Y, lengthv, width, stride, fp32_func_param0, fp32_func_param1, fp32_func_param2);
	else if (fp32_func_type == FP32_Func_Sigmoid)   __temp_affine2D_row_with_function<FP32_Func_Sigmoid>  (stream, X, A, B, row_lengthv, Y, lengthv, width, stride, fp32_func_param0, fp32_func_param1, fp32_func_param2);
	else if (fp32_func_type == FP32_Func_Tanh)      __temp_affine2D_row_with_function<FP32_Func_Tanh>     (stream, X, A, B, row_lengthv, Y, lengthv, width, stride, fp32_func_param0, fp32_func_param1, fp32_func_param2);
	else throwException(env, "Unknown FP32_Func_Type");
}

#endif

#endif