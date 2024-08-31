#pragma once

#ifndef LINEAR_DUAL_2D_WITH_FUNCTION_H
#define LINEAR_DUAL_2D_WITH_FUNCTION_H

//lengthv = height * stride
//stride = (width + 3)/4*4
//[lenghtv, stride] % 4 == 0
#ifndef LINEAR_DUAL_2D_WITH_FUNCTION_CALL
#define LINEAR_DUAL_2D_WITH_FUNCTION_CALL

//LB = log2(BLOCK_SIZE)
//LT = log2(THREAD_ELEMENT_VISIT_SIZE)

//lengthv < 256
#define linear_dual2d_with_function_k4_small(stream, X1, X2, alpha, beta, gamma, Y, lengthv, width, stride)\
	linear_dual2D_with_function_kernel_4<fp32_func_type>\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(X1, X2, alpha, beta, gamma, Y, lengthv, width, stride,\
				fp32_func_param0, fp32_func_param1, fp32_func_param2)

//common
#define linear_dual2d_with_function_k4(stream, LB, LT, X1, X2, alpha, beta, gamma, Y, lengthv, width, stride)\
	linear_dual2D_with_function_kernel_4<fp32_func_type>\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(X1, X2, alpha, beta, gamma, Y, lengthv, width, stride,\
				fp32_func_param0, fp32_func_param1, fp32_func_param2)

//lengthv < 256
#define linear_dual2d_with_function_k4_max(stream, X1, X2, alpha, beta, gamma, Y, lengthv, width, stride)\
	linear_dual2D_with_function_kernel_4<fp32_func_type>\
		<<< GRID_MAX, 32, 0, stream >>>\
			(X1, X2, alpha, beta, gamma, Y, lengthv, width, stride,\
				fp32_func_param0, fp32_func_param1, fp32_func_param2)

#endif


#ifndef LINEAR_DUAL_2D_WITH_FUNCTION_KERNEL
#define LINEAR_DUAL_2D_WITH_FUNCTION_KERNEL

//forward propagation:
//(1) Y1 = alpha*X1 + beta*X2 + gamma
//(2) Y2 = function(Y1)

template<int fp32_func_type>
__global__ void linear_dual2D_with_function_kernel_4(
	const float* __restrict__ X1,
	const float* __restrict__ X2, 
	float alpha, float beta, float gamma, 
	      float* __restrict__ Y,
	int lengthv, int width, int stride,
	float fp32_func_param0, float fp32_func_param1, float fp32_func_param2)
{
	const int step = gridDim.x*blockDim.x, step4 = step << 2;
	const int index = blockIdx.x*blockDim.x + threadIdx.x;

	for (int index4 = index << 2; index4 < lengthv; index4 += step4) 
	{
		float4 x1 = *(float4*)(X1 + index4);
		float4 x2 = *(float4*)(X2 + index4);

		float4 y1;//Y1 = alpha*X1 + beta*X2 + gamma
		y1.x = alpha * x1.x + beta * x2.x + gamma;
		y1.y = alpha * x1.y + beta * x2.y + gamma;
		y1.z = alpha * x1.z + beta * x2.z + gamma;
		y1.w = alpha * x1.w + beta * x2.w + gamma;

		float4 y2;//Y2 = function(Y1)
		y2.x = fp32_func_forward<fp32_func_type>(y1.x, fp32_func_param0, fp32_func_param1, fp32_func_param2);
		y2.y = fp32_func_forward<fp32_func_type>(y1.y, fp32_func_param0, fp32_func_param1, fp32_func_param2);
		y2.z = fp32_func_forward<fp32_func_type>(y1.z, fp32_func_param0, fp32_func_param1, fp32_func_param2);
		y2.w = fp32_func_forward<fp32_func_type>(y1.w, fp32_func_param0, fp32_func_param1, fp32_func_param2);

		within_width(y2, index4, stride, width);
		*(float4*)(Y + index4) = y2;
	}
}

#endif


#ifndef LINEAR_DUAL_2D_WITH_FUNCTION
#define LINEAR_DUAL_2D_WITH_FUNCTION

template<int fp32_func_type>
void __temp_linear_dual2D_with_function(cudaStream_t stream,
	const float* X1, const float* X2,
	float alpha, float beta, float gamma,
	      float* Y,
	int lengthv, int width, int stride,
	float fp32_func_param0, float fp32_func_param1, float fp32_func_param2)
{
	if (lengthv < 256) { linear_dual2d_with_function_k4_small(stream, X1, X2, alpha, beta, gamma, Y, lengthv, width, stride); return; }
	if (lengthv > LENGTHV_MAX) { linear_dual2d_with_function_k4_max(stream, X1, X2, alpha, beta, gamma, Y, lengthv, width, stride); return; }
	linear_dual2d_with_function_k4(stream, 5, 2, X1, X2, alpha, beta, gamma, Y, lengthv, width, stride);
}


void __linear_dual2D_with_function(JNIEnv* env, cudaStream_t stream,
	const float* X1, const float* X2,
	float alpha, float beta, float gamma,
	      float* Y,
	int lengthv, int width, int stride,
	int fp32_func_type, float fp32_func_param0, float fp32_func_param1, float fp32_func_param2)
{
	if      (fp32_func_type == FP32_Func_Relu)      __temp_linear_dual2D_with_function<FP32_Func_Relu>     (stream, X1, X2, alpha, beta, gamma, Y, lengthv, width, stride, fp32_func_param0, fp32_func_param1, fp32_func_param2);
	else if (fp32_func_type == FP32_Func_LeakyRelu) __temp_linear_dual2D_with_function<FP32_Func_LeakyRelu>(stream, X1, X2, alpha, beta, gamma, Y, lengthv, width, stride, fp32_func_param0, fp32_func_param1, fp32_func_param2);
	else if (fp32_func_type == FP32_Func_Elu)       __temp_linear_dual2D_with_function<FP32_Func_Elu>      (stream, X1, X2, alpha, beta, gamma, Y, lengthv, width, stride, fp32_func_param0, fp32_func_param1, fp32_func_param2);
	else if (fp32_func_type == FP32_Func_Softplus)  __temp_linear_dual2D_with_function<FP32_Func_Softplus> (stream, X1, X2, alpha, beta, gamma, Y, lengthv, width, stride, fp32_func_param0, fp32_func_param1, fp32_func_param2);
	else if (fp32_func_type == FP32_Func_Gelu)      __temp_linear_dual2D_with_function<FP32_Func_Gelu>     (stream, X1, X2, alpha, beta, gamma, Y, lengthv, width, stride, fp32_func_param0, fp32_func_param1, fp32_func_param2);
	else if (fp32_func_type == FP32_Func_Sigmoid)   __temp_linear_dual2D_with_function<FP32_Func_Sigmoid>  (stream, X1, X2, alpha, beta, gamma, Y, lengthv, width, stride, fp32_func_param0, fp32_func_param1, fp32_func_param2);
	else if (fp32_func_type == FP32_Func_Tanh)      __temp_linear_dual2D_with_function<FP32_Func_Tanh>     (stream, X1, X2, alpha, beta, gamma, Y, lengthv, width, stride, fp32_func_param0, fp32_func_param1, fp32_func_param2);
	else throwException(env, "Unknown FP32_Func_Type");
}

#endif

#endif