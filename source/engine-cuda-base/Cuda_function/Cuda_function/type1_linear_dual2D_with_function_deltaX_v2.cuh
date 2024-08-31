#pragma once

#ifndef LINEAR_DUAL_2D_WITH_FUNCTION_DELTAX_V2_H
#define LINEAR_DUAL_2D_WITH_FUNCTION_DELTAX_V2_H

//lengthv = height * stride
//stride = (width + 3)/4*4
//[lenghtv, stride] % 4 == 0
//V2: holdX(), {X1, X2} are not changed
#ifndef LINEAR_DUAL_2D_WITH_FUNCTION_DELTAX_V2_CALL
#define LINEAR_DUAL_2D_WITH_FUNCTION_DELTAX_V2_CALL

//LB = log2(BLOCK_SIZE)
//LT = log2(THREAD_ELEMENT_VISIT_SIZE)

//lengthv < 256
#define linear_dual2d_with_function_deltaX_v2_k4_small(stream, deltaX1, deltaX2, deltaY, X1, X2, alpha, beta, gamma, lengthv)\
	linear_dual2D_with_function_deltaX_v2_kernel_4<fp32_func_type>\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(deltaX1, deltaX2, deltaY, X1, X2, alpha, beta, gamma, lengthv, width, stride,\
				fp32_func_param0, fp32_func_param1, fp32_func_param2)

//common
#define linear_dual2d_with_function_deltaX_v2_k4(stream, LB, LT, deltaX1, deltaX2, deltaY, X1, X2, alpha, beta, gamma, lengthv)\
	linear_dual2D_with_function_deltaX_v2_kernel_4<fp32_func_type>\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(deltaX1, deltaX2, deltaY, X1, X2, alpha, beta, gamma, lengthv, width, stride,\
				fp32_func_param0, fp32_func_param1, fp32_func_param2)

//lengthv > lengthv_max
#define linear_dual2d_with_function_deltaX_v2_k4_max(stream, deltaX1, deltaX2, deltaY, X1, X2, alpha, beta, gamma, lengthv)\
	linear_dual2D_with_function_deltaX_v2_kernel_4<fp32_func_type>\
		<<< GRID_MAX, 32, 0, stream >>>\
			(deltaX1, deltaX2, deltaY, X1, X2, alpha, beta, gamma, lengthv, width, stride,\
				fp32_func_param0, fp32_func_param1, fp32_func_param2)

#endif


#ifndef LINEAR_DUAL_2D_WITH_FUNCTION_DELTAX_V2_KERNEL
#define LINEAR_DUAL_2D_WITH_FUNCTION_DELTAX_V2_KERNEL

//forward propagation:
//(1) Y1 = alpha*X1 + beta*X2 + gamma
//(2) Y2 = function(Y1)
//backward propagation:
//(1) Y1 = alpha*X1 + beta*X2 + gamma 
//(2) deltaY1 = deltaY2 * function'(Y1)
//(3) deltaX1 = deltaY1 * alpha
//(4) deltaX2 = deltaY2 * beta

template<int fp32_func_type>
__global__ void linear_dual2D_with_function_deltaX_v2_kernel_4(
	      float* __restrict__ deltaX1,
	      float* __restrict__ deltaX2,
	const float* __restrict__ deltaY,
	const float* __restrict__ X1,
	const float* __restrict__ X2,
	float alpha, float beta, float gamma,
	int lengthv, int width, int stride,
	float fp32_func_param0, float fp32_func_param1, float fp32_func_param2)
{
	const int step = gridDim.x*blockDim.x, step4 = step << 2;
	const int index = blockIdx.x*blockDim.x + threadIdx.x;

	fp32_func* func = new_fp32_func(fp32_func_type, fp32_func_param0, fp32_func_param1, fp32_func_param2);
	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		float4 dy2 = *(float4*)(deltaY + index4);
		float4 x1 = *(float4*)(X1 + index4);
		float4 x2 = *(float4*)(X2 + index4);

		float4 y1;//Y1 = alpha*X1 + beta*X2 + gamma
		y1.x = alpha * x1.x + beta * x2.x + gamma;
		y1.y = alpha * x1.y + beta * x2.y + gamma;
		y1.z = alpha * x1.z + beta * x2.z + gamma;
		y1.w = alpha * x1.w + beta * x2.w + gamma;

		float4 dy1;//deltaY1 = deltaY2 * function'(Y1)
		dy1.x = dy2.x * fp32_func_derivative_v2<fp32_func_type>(y1.x, fp32_func_param0, fp32_func_param1, fp32_func_param2);
		dy1.y = dy2.y * fp32_func_derivative_v2<fp32_func_type>(y1.y, fp32_func_param0, fp32_func_param1, fp32_func_param2);
		dy1.z = dy2.z * fp32_func_derivative_v2<fp32_func_type>(y1.z, fp32_func_param0, fp32_func_param1, fp32_func_param2);
		dy1.w = dy2.w * fp32_func_derivative_v2<fp32_func_type>(y1.w, fp32_func_param0, fp32_func_param1, fp32_func_param2);

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
	delete func;
}

#endif


#ifndef LINEAR_DUAL_2D_WITH_FUNCTION_DELTAX_V2
#define LINEAR_DUAL_2D_WITH_FUNCTION_DELTAX_V2

template<int fp32_func_type>
void __temp_linear_dual2D_with_function_deltaX_v2(cudaStream_t stream,
	      float* deltaX1, float* deltaX2,
	const float* deltaY,
	const float* X1, const float* X2,
	float alpha, float beta, float gamma, 
	int lengthv, int width, int stride,
	float fp32_func_param0, float fp32_func_param1, float fp32_func_param2)
{
	if (lengthv < 256) { linear_dual2d_with_function_deltaX_v2_k4_small(stream, deltaX1, deltaX2, deltaY, X1, X2, alpha, beta, gamma, lengthv); return; }
	if (lengthv > LENGTHV_MAX) { linear_dual2d_with_function_deltaX_v2_k4_max(stream, deltaX1, deltaX2, deltaY, X1, X2, alpha, beta, gamma, lengthv); return; }
	linear_dual2d_with_function_deltaX_v2_k4(stream, 5, 2, deltaX1, deltaX2, deltaY, X1, X2, alpha, beta, gamma, lengthv);
}


void __linear_dual2D_with_function_deltaX_v2(JNIEnv* env, cudaStream_t stream,
	      float* deltaX1, float* deltaX2,
	const float* deltaY,
	const float* X1, const float* X2,
	float alpha, float beta, float gamma,
	int lengthv, int width, int stride,
	int fp32_func_type, float fp32_func_param0, float fp32_func_param1, float fp32_func_param2)
{
	if (fp32_func_type == FP32_Func_Relu)           __temp_linear_dual2D_with_function_deltaX_v2<FP32_Func_Relu>     (stream, deltaX1, deltaX2, deltaY, X1, X2, alpha, beta, gamma, lengthv, width, stride, fp32_func_param0, fp32_func_param1, fp32_func_param2);
	else if (fp32_func_type == FP32_Func_LeakyRelu) __temp_linear_dual2D_with_function_deltaX_v2<FP32_Func_LeakyRelu>(stream, deltaX1, deltaX2, deltaY, X1, X2, alpha, beta, gamma, lengthv, width, stride, fp32_func_param0, fp32_func_param1, fp32_func_param2);
	else if (fp32_func_type == FP32_Func_Elu)       __temp_linear_dual2D_with_function_deltaX_v2<FP32_Func_Elu>      (stream, deltaX1, deltaX2, deltaY, X1, X2, alpha, beta, gamma, lengthv, width, stride, fp32_func_param0, fp32_func_param1, fp32_func_param2);
	else if (fp32_func_type == FP32_Func_Softplus)  __temp_linear_dual2D_with_function_deltaX_v2<FP32_Func_Softplus> (stream, deltaX1, deltaX2, deltaY, X1, X2, alpha, beta, gamma, lengthv, width, stride, fp32_func_param0, fp32_func_param1, fp32_func_param2);
	else if (fp32_func_type == FP32_Func_Gelu)      __temp_linear_dual2D_with_function_deltaX_v2<FP32_Func_Gelu>     (stream, deltaX1, deltaX2, deltaY, X1, X2, alpha, beta, gamma, lengthv, width, stride, fp32_func_param0, fp32_func_param1, fp32_func_param2);
	else if (fp32_func_type == FP32_Func_Sigmoid)   __temp_linear_dual2D_with_function_deltaX_v2<FP32_Func_Sigmoid>  (stream, deltaX1, deltaX2, deltaY, X1, X2, alpha, beta, gamma, lengthv, width, stride, fp32_func_param0, fp32_func_param1, fp32_func_param2);
	else if (fp32_func_type == FP32_Func_Tanh)      __temp_linear_dual2D_with_function_deltaX_v2<FP32_Func_Tanh>     (stream, deltaX1, deltaX2, deltaY, X1, X2, alpha, beta, gamma, lengthv, width, stride, fp32_func_param0, fp32_func_param1, fp32_func_param2);
	else throwException(env, "Unknown FP32_Func_Type");

}

#endif

#endif