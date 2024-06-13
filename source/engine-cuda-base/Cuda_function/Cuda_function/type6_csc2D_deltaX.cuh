#pragma once

#ifndef CSC_2D_DELTAX_H
#define CSC_2D_DELTAX_H

//lengthv = height * stride
//stride = (width + 3)/4*4
//[lenghtv, stride] % 4 == 0
#ifndef CSC_2D_DELTAX_CALL
#define CSC_2D_DELTAX_CALL

//lengthv < 256
#define csc2d_deltaX_k4_small(stream,  deltaX, deltaY, X, alpha, beta, lengthv, width, stride)\
	csc2D_deltaX_kernel_4\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(deltaX, deltaY, X, alpha, beta, lengthv, width, stride)

//common
#define csc2d_deltaX_k4(stream, LB, LT, deltaX, deltaY, X, alpha, beta, lengthv, width, stride)\
	csc2D_deltaX_kernel_4\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(deltaX, deltaY, X, alpha, beta, lengthv, width, stride)

//lengthv > lengthv_max
#define csc2d_deltaX_k4_max(stream,  deltaX, deltaY, X, alpha, beta, lengthv, width, stride)\
	csc2D_deltaX_kernel_4\
		<<< GRID_MAX, 32, 0, stream >>>\
			(deltaX, deltaY, X, alpha, beta, lengthv, width, stride)

#endif


#ifndef CSC_2D_DELTAX_KERNEL
#define CSC_2D_DELTAX_KERNEL

//<1> Y = csc(alpha*X + beta) = 1 / sin(alpha*X + beta)
//    Y = sec(alpha*X + beta) = csc(alpha*X + (beta + 0.5pi))
//<2> Y' = { -alpha * cos(alpha*X + beta) } / sin(alpha*X + beta)^2
//    Y' = { -alpha * cos(alpha*X + beta) } * Y^2
//STEP: 
//<1> X = alpha*X + beta
//<2> Y = sin(X)
//<3> Y' = -alpha*cos(X) / Y^2
__global__ void csc2D_deltaX_kernel_4(
	      float* __restrict__ deltaX,
	const float* __restrict__ deltaY,
	const float* __restrict__ X,
	float alpha, float beta,
	int lengthv, int width, int stride)
{
	int step = gridDim.x*blockDim.x, step4 = step << 2;
	int index = blockIdx.x*blockDim.x + threadIdx.x;

	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		float4 x = *(float4*)(X + index4);
		float4 dy = *(float4*)(deltaY + index4);

		simdLinear4(x, alpha, x, beta);//<1> X = alpha*X + beta

		float4 y;//<2> Y = sin(X)
		y.x = sinf(x.x);
		y.y = sinf(x.y);
		y.z = sinf(x.z);
		y.w = sinf(x.w);

		float4 dx;//<3> Y' = -alpha*cos(X) / Y^2
		dx.x = -alpha * cosf(x.x) / (y.x * y.x);
		dx.y = -alpha * cosf(x.y) / (y.y * y.y);
		dx.z = -alpha * cosf(x.z) / (y.z * y.z);
		dx.w = -alpha * cosf(x.w) / (y.w * y.w);

		simdMul4(dx, dx, dy);//deltaX = deltaY * driY

		within_width(dx, index4, stride, width);
		*(float4*)(deltaX + index4) = dx;
	}
}

#endif


void __csc2D_deltaX(cudaStream_t stream,
	      float* deltaX,
	const float* deltaY,
	const float *X, float alpha, float beta,
	int lengthv, int width, int stride)
{
	if (lengthv < 256) { csc2d_deltaX_k4_small(stream, deltaX, deltaY, X, alpha, beta, lengthv, width, stride); return; }
	if (lengthv > LENGTHV_MAX) { csc2d_deltaX_k4_max(stream, deltaX, deltaY, X, alpha, beta, lengthv, width, stride); return; }
	csc2d_deltaX_k4(stream, 5, 2, deltaX, deltaY, X, alpha, beta, lengthv, width, stride);
}

#endif