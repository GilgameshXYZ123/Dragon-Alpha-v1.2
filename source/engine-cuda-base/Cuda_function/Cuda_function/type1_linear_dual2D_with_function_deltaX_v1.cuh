#pragma once

#ifndef LINEAR_DUAL_2D_WITH_FUNCTION_DELTAX_V1_H
#define LINEAR_DUAL_2D_WITH_FUNCTION_DELTAX_V1_H

//lengthv = height * stride
//stride = (width + 3)/4*4
//[lenghtv, stride] % 4 == 0
//V1: holdY(), Y is not changed
#ifndef LINEAR_DUAL_2D_WITH_FUNCTION_DELTAX_V1_CALL
#define LINEAR_DUAL_2D_WITH_FUNCTION_DELTAX_V1_CALL

//LB = log2(BLOCK_SIZE)
//LT = log2(THREAD_ELEMENT_VISIT_SIZE)

//lengthv < 256
#define linear_dual2d_with_function_deltaX_v1_k4_small(stream, deltaX1, deltaX2, deltaY, Y, alpha, beta, lengthv)\
	linear_dual2D_with_function_deltaX_v1_kernel_4<fp32_func_type>\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(deltaX1, deltaX2, deltaY, Y, alpha, beta, lengthv, width, stride,\
				fp32_func_param0, fp32_func_param1, fp32_func_param2)

//common
#define linear_dual2d_with_function_deltaX_v1_k4(stream, LB, LT, deltaX1, deltaX2, deltaY, Y, alpha, beta, lengthv)\
	linear_dual2D_with_function_deltaX_v1_kernel_4<fp32_func_type>\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(deltaX1, deltaX2, deltaY, Y, alpha, beta, lengthv, width, stride,\
				fp32_func_param0, fp32_func_param1, fp32_func_param2)

//lengthv > lengthv_max
#define linear_dual2d_with_function_deltaX_v1_k4_max(stream, deltaX1, deltaX2, deltaY, Y, alpha, beta, lengthv)\
	linear_dual2D_with_function_deltaX_v1_kernel_4<fp32_func_type>\
		<<< GRID_MAX, 32, 0, stream >>>\
			(deltaX1, deltaX2, deltaY, Y, alpha, beta, lengthv, width, stride,\
				fp32_func_param0, fp32_func_param1, fp32_func_param2)

#endif


#ifndef LINEAR_DUAL_2D_WITH_FUNCTION_DELTAX_V1_KERNEL
#define LINEAR_DUAL_2D_WITH_FUNCTION_DELTAX_V1_KERNEL

//forward propagation:
//(1) Y1 = alpha*X1 + beta*X2 + gamma
//(2) Y2 = function(Y1)
//backward propagation:
//(2) deltaY1 = deltaY2 * function'(Y2)
//(3) deltaX1 = deltaY1 * alpha
//(4) deltaX2 = deltaY2 * beta

template<int fp32_func_type>
__global__ void linear_dual2D_with_function_deltaX_v1_kernel_4(
	      float* __restrict__ deltaX1,
	      float* __restrict__ deltaX2,
	const float* __restrict__ deltaY,
	const float* __restrict__ Y,
	float alpha, float beta,
	int lengthv, int width, int stride,
	float fp32_func_param0, float fp32_func_param1, float fp32_func_param2)
{
	const int step = gridDim.x*blockDim.x, step4 = step << 2;
	const int index = blockIdx.x*blockDim.x + threadIdx.x;

	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		float4 dy2 = *(float4*)(deltaY + index4);
		float4 y2 = *(float4*)(Y + index4);

		float4 dy1;//deltaY1 = deltaY2 * function'(Y2)
		dy1.x = dy2.x * fp32_func_derivative_v1<fp32_func_type>(y2.x, fp32_func_param0, fp32_func_param1, fp32_func_param2);
		dy1.y = dy2.y * fp32_func_derivative_v1<fp32_func_type>(y2.y, fp32_func_param0, fp32_func_param1, fp32_func_param2);
		dy1.z = dy2.z * fp32_func_derivative_v1<fp32_func_type>(y2.z, fp32_func_param0, fp32_func_param1, fp32_func_param2);
		dy1.w = dy2.w * fp32_func_derivative_v1<fp32_func_type>(y2.w, fp32_func_param0, fp32_func_param1, fp32_func_param2);

		float4 dx1, dx2;//deltaX1 = deltaY1 * alpha; deltaX2 = deltaY2 * beta
		dx1.x = dy1.x * alpha; dx2.x = dy1.x * beta;
		dx1.y = dy1.y * alpha; dx2.y = dy1.y * beta;
		dx1.z = dy1.z * alpha; dx2.z = dy1.z * beta;
		dx1.w = dy1.w * alpha; dx2.w = dy1.w * beta;

		within_width(dx1, index4, stride, width);
		within_width(dx2, index4, stride, width);
		*(float4*)(deltaX1 + index4) = dx1;
		*(float4*)(deltaX2 + index4) = dx2;
	}
}

#endif


#ifndef LINEAR_DUAL_2D_WITH_FUNCTION_DELTAX_V1
#define LINEAR_DUAL_2D_WITH_FUNCTION_DELTAX_V1

template<int fp32_func_type>
void __temp_linear_dual2D_with_function_deltaX_v1(cudaStream_t stream,
	      float* deltaX1, float* deltaX2,
	const float* deltaY,
	const float* Y, 
	float alpha, float beta,
	int lengthv, int width, int stride,
	float fp32_func_param0, float fp32_func_param1, float fp32_func_param2)
{
	if (lengthv < 256) { linear_dual2d_with_function_deltaX_v1_k4_small(stream, deltaX1, deltaX2, deltaY, Y, alpha, beta, lengthv); return; }
	if (lengthv > LENGTHV_MAX) { linear_dual2d_with_function_deltaX_v1_k4_max(stream, deltaX1, deltaX2, deltaY, Y, alpha, beta, lengthv); return; }
	linear_dual2d_with_function_deltaX_v1_k4(stream, 5, 2, deltaX1, deltaX2, deltaY, Y, alpha, beta, lengthv);
}


void __linear_dual2D_with_function_deltaX_v1(JNIEnv* env, cudaStream_t stream,
	      float* deltaX1, float* deltaX2,
	const float* deltaY,
	const float* Y,
	float alpha, float beta, 
	int lengthv, int width, int stride,
	int fp32_func_type, float fp32_func_param0, float fp32_func_param1, float fp32_func_param2)
{
	if      (fp32_func_type == FP32_Func_Relu)      __temp_linear_dual2D_with_function_deltaX_v1<FP32_Func_Relu>     (stream, deltaX1, deltaX2, deltaY, Y, alpha, beta, lengthv, width, stride, fp32_func_param0, fp32_func_param1, fp32_func_param2);
	else if (fp32_func_type == FP32_Func_LeakyRelu) __temp_linear_dual2D_with_function_deltaX_v1<FP32_Func_LeakyRelu>(stream, deltaX1, deltaX2, deltaY, Y, alpha, beta, lengthv, width, stride, fp32_func_param0, fp32_func_param1, fp32_func_param2);
	else if (fp32_func_type == FP32_Func_Elu)       __temp_linear_dual2D_with_function_deltaX_v1<FP32_Func_Elu>      (stream, deltaX1, deltaX2, deltaY, Y, alpha, beta, lengthv, width, stride, fp32_func_param0, fp32_func_param1, fp32_func_param2);
	else if (fp32_func_type == FP32_Func_Softplus)  __temp_linear_dual2D_with_function_deltaX_v1<FP32_Func_Softplus> (stream, deltaX1, deltaX2, deltaY, Y, alpha, beta, lengthv, width, stride, fp32_func_param0, fp32_func_param1, fp32_func_param2);
	else if (fp32_func_type == FP32_Func_Gelu)      __temp_linear_dual2D_with_function_deltaX_v1<FP32_Func_Gelu>     (stream, deltaX1, deltaX2, deltaY, Y, alpha, beta, lengthv, width, stride, fp32_func_param0, fp32_func_param1, fp32_func_param2);
	else if (fp32_func_type == FP32_Func_Sigmoid)   __temp_linear_dual2D_with_function_deltaX_v1<FP32_Func_Sigmoid>  (stream, deltaX1, deltaX2, deltaY, Y, alpha, beta, lengthv, width, stride, fp32_func_param0, fp32_func_param1, fp32_func_param2);
	else if (fp32_func_type == FP32_Func_Tanh)      __temp_linear_dual2D_with_function_deltaX_v1<FP32_Func_Tanh>     (stream, deltaX1, deltaX2, deltaY, Y, alpha, beta, lengthv, width, stride, fp32_func_param0, fp32_func_param1, fp32_func_param2);
	else throwException(env, "Unknown FP32_Func_Type");
}

#endif

#endif