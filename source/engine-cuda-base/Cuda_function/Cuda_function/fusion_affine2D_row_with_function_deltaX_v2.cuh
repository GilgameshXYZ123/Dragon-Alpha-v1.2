#pragma once

#ifndef AFFINE_2D_ROW_WITH_FUNCTION_DELTAX_V2_H
#define AFFINE_2D_ROW_WITH_FUNCTION_DELTAX_V2_H

//lengthv = height * stride
//stride = (width + 3)/4*4
//[lenghtv, stride] % 4 == 0
//V2: holdX(), X is not changed
#ifndef AFFINE_2D_ROW_WITH_FUNCTION_DELTAX_V2_CALL
#define AFFINE_2D_ROW_WITH_FUNCTION_DELTAX_V2_CALL

//LB = log2(BLOCK_SIZE)
//LT = log2(THREAD_ELEMENT_VISIT_SIZE)

//lengthv < 256
#define affine2d_row_with_function_deltaX_v2_k4_small(stream, deltaX, deltaY, X, A, B, row_lengthv, lengthv, width, stride)\
	affine2D_row_with_function_deltaX_v2_kernel_4<fp32_func_type>\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(deltaX, deltaY, X, A, B, row_lengthv, lengthv, width, stride,\
				fp32_func_param0, fp32_func_param1, fp32_func_param2)

//common
#define affine2d_row_with_function_deltaX_v2_k4(stream, LB, LT, deltaX, deltaY, X, A, B, row_lengthv, lengthv, width, stride)\
	affine2D_row_with_function_deltaX_v2_kernel_4<fp32_func_type>\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(deltaX, deltaY, X, A, B, row_lengthv, lengthv, width, stride,\
				fp32_func_param0, fp32_func_param1, fp32_func_param2)

//lengthv > lengthv_max
#define affine2d_row_with_function_deltaX_v2_k4_max(stream, deltaX, deltaY, X, A, B, row_lengthv, lengthv, width, stride)\
	affine2D_row_with_function_deltaX_v2_kernel_4<fp32_func_type>\
		<<< GRID_MAX, 32, 0, stream >>>\
			(deltaX, deltaY, X, A, B, row_lengthv, lengthv, width, stride,\
				fp32_func_param0, fp32_func_param1, fp32_func_param2)

#endif


#ifndef AFFINE_2D_ROW_WITH_FUNCTION_DELTAX_V2_KERNEL
#define AFFINE_2D_ROW_WITH_FUNCTION_DELTAX_V2_KERNEL

//Forward propagation:
//(1) Y1 = A*X + B
//(2) Y2 = function(Y1)
//
//Backward propagation:
//(1) Y1 = A*X + B
//(2) deltaY1 = deltaY2 * derivative_v2(Y1), deltaY2 = deltaY
//(3) deltaX = A * deltaY1

template<int fp32_func_type>
__global__ void affine2D_row_with_function_deltaX_v2_kernel_4(
	      float* __restrict__ deltaX,
	const float* __restrict__ deltaY,
	const float* __restrict__ X,
	const float* __restrict__ A,
	const float* __restrict__ B, int row_lengthv,
	int lengthv, int width, int stride,
	float fp32_func_param0, float fp32_func_param1, float fp32_func_param2)
{
	const int step = gridDim.x*blockDim.x, step4 = step << 2;
	const int index = blockIdx.x*blockDim.x + threadIdx.x;

	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		const float4 x = *(float4*)(X + index4);
		const float4 dy2 = *(float4*)(deltaY + index4);

		const int field_index4 = index4 % row_lengthv;
		const float4 a = *(float4*)(A + field_index4);
		const float4 b = *(float4*)(B + field_index4);

		float4 y1;//Y1 = A*X + B
		y1.x = a.x * x.x + b.x;
		y1.y = a.y * x.y + b.y;
		y1.z = a.z * x.z + b.z;
		y1.w = a.w * x.w + b.w;

		float4 dy1;//deltaY1 = deltaY2 * derivative_v2(Y1)
		dy1.x = dy2.x * fp32_func_derivative_v2<fp32_func_type>(y1.x, fp32_func_param0, fp32_func_param1, fp32_func_param2);
		dy1.y = dy2.y * fp32_func_derivative_v2<fp32_func_type>(y1.y, fp32_func_param0, fp32_func_param1, fp32_func_param2);
		dy1.z = dy2.z * fp32_func_derivative_v2<fp32_func_type>(y1.z, fp32_func_param0, fp32_func_param1, fp32_func_param2);
		dy1.w = dy2.w * fp32_func_derivative_v2<fp32_func_type>(y1.w, fp32_func_param0, fp32_func_param1, fp32_func_param2);

		float4 dx;//deltaX = A * deltaY1
		dx.x = a.x * dy1.x;
		dx.y = a.y * dy1.y;
		dx.z = a.z * dy1.z;
		dx.w = a.w * dy1.w;

		within_width(dx, index4, stride, width);
		*(float4*)(deltaX + index4) = dx;
	}
}

#endif


#ifndef AFFINE_2D_ROW_WITH_FUNCTION_DELTAX_V2
#define AFFINE_2D_ROW_WITH_FUNCTION_DELTAX_V2

template<int fp32_func_type>
void __temp_affine2D_row_with_function_deltaX_v2(cudaStream_t stream,
	      float* deltaX,
	const float* deltaY,
	const float* X,
	const float* A, const float* B,
	int row_lengthv,
	int lengthv, int width, int stride,
	float fp32_func_param0, float fp32_func_param1, float fp32_func_param2)
{
	if (lengthv < 256) { affine2d_row_with_function_deltaX_v2_k4_small(stream, deltaX, deltaY, X, A, B, row_lengthv, lengthv, width, stride); return; }
	if (lengthv > LENGTHV_MAX) { affine2d_row_with_function_deltaX_v2_k4_max(stream, deltaX, deltaY, X, A, B, row_lengthv, lengthv, width, stride); return; }
	affine2d_row_with_function_deltaX_v2_k4(stream, 5, 2, deltaX, deltaY, X, A, B, row_lengthv, lengthv, width, stride);
}


void __affine2D_row_with_function_deltaX_v2(JNIEnv* env, cudaStream_t stream,
	      float* deltaX,
	const float* deltaY,
	const float* X,
	const float* A, const float* B,
	int row_lengthv,
	int lengthv, int width, int stride,
	int fp32_func_type, float fp32_func_param0, float fp32_func_param1, float fp32_func_param2)
{
	if      (fp32_func_type == FP32_Func_Relu)      __temp_affine2D_row_with_function_deltaX_v2<FP32_Func_Relu>     (stream, deltaX, deltaY, X, A, B, row_lengthv, lengthv, width, stride, fp32_func_param0, fp32_func_param1, fp32_func_param2);
	else if (fp32_func_type == FP32_Func_LeakyRelu) __temp_affine2D_row_with_function_deltaX_v2<FP32_Func_LeakyRelu>(stream, deltaX, deltaY, X, A, B, row_lengthv, lengthv, width, stride, fp32_func_param0, fp32_func_param1, fp32_func_param2);
	else if (fp32_func_type == FP32_Func_Elu)       __temp_affine2D_row_with_function_deltaX_v2<FP32_Func_Elu>      (stream, deltaX, deltaY, X, A, B, row_lengthv, lengthv, width, stride, fp32_func_param0, fp32_func_param1, fp32_func_param2);
	else if (fp32_func_type == FP32_Func_Softplus)  __temp_affine2D_row_with_function_deltaX_v2<FP32_Func_Softplus> (stream, deltaX, deltaY, X, A, B, row_lengthv, lengthv, width, stride, fp32_func_param0, fp32_func_param1, fp32_func_param2);
	else if (fp32_func_type == FP32_Func_Gelu)      __temp_affine2D_row_with_function_deltaX_v2<FP32_Func_Gelu>     (stream, deltaX, deltaY, X, A, B, row_lengthv, lengthv, width, stride, fp32_func_param0, fp32_func_param1, fp32_func_param2);
	else if (fp32_func_type == FP32_Func_Sigmoid)   __temp_affine2D_row_with_function_deltaX_v2<FP32_Func_Sigmoid>  (stream, deltaX, deltaY, X, A, B, row_lengthv, lengthv, width, stride, fp32_func_param0, fp32_func_param1, fp32_func_param2);
	else if (fp32_func_type == FP32_Func_Tanh)      __temp_affine2D_row_with_function_deltaX_v2<FP32_Func_Tanh>     (stream, deltaX, deltaY, X, A, B, row_lengthv, lengthv, width, stride, fp32_func_param0, fp32_func_param1, fp32_func_param2);
	else throwException(env, "Unknown FP32_Func_Type");
}

#endif

#endif