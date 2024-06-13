#pragma once

#ifndef LINEAR_BOUND_SWITCH_MUL_2D_H
#define LINEAR_BOUND_SWITCH_MUL_2D_H

//lengthv = height * stride
//stride = (width + 3)/4*4
//[lenghtv, stride] % 4 == 0
#ifndef LINEAR_BOUND_SWITCH_MUL_2D_CALL
#define LINEAR_BOUND_SWITCH_MUL_2D_CALL

//LB = log2(BLOCK_SIZE)
//LT = log2(THREAD_ELEMENT_VISIT_SIZE)

//lengthv < 256
#define linear_bound_switch_mul2d_k4_small(stream, alpha, X1, vmin, vmax, X2, v1, v2, v3, Y, lengthv, width, stride)\
	linear_bound_switch_mul2D_kernel_4\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(alpha, X1, vmin, vmax, X2, v1, v2, v3, Y, lengthv, width, stride)

//common
#define linear_bound_switch_mul2d_k4(stream, LB, LT, alpha, X1, vmin, vmax, X2, v1, v2, v3, Y, lengthv, width, stride)\
	linear_bound_switch_mul2D_kernel_4\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(alpha, X1, vmin, vmax, X2, v1, v2, v3, Y, lengthv, width, stride)

//lengthv > lengthv_max
#define linear_bound_switch_mul2d_k4_max(stream, alpha, X1, vmin, vmax, X2, v1, v2, v3, Y, lengthv, width, stride)\
	linear_bound_switch_mul2D_kernel_4\
		<<< GRID_MAX, 32, 0, stream >>>\
			(alpha, X1, vmin, vmax, X2, v1, v2, v3, Y, lengthv, width, stride)

#endif


#ifndef LINEAR_BOUND_SWITCH_MUL_2D_KERNEL
#define LINEAR_BOUND_SWITCH_MUL_2D_KERNEL

//flag1 | flag2 | flag3
//(1) Y = X2 * v3, (alpha * X1) >= vmax,       [vmax, +inf)
//(2) Y = X2 * v2, vmin < (alpha * X1) < vmax, (vmin, vmax) 
//(3) Y = X2 * v1, (alpha * X1) <= vmin,       (-inf, vmin]
//Step:
//<1> flag1 = (alpha * X1) <= vmin
//<2> flag3 = (alpha * X1) >= vmax
//<3> flag2 = { vmin < (alpha * X1) < vmax } = !flag1 && !flag3 = !(flag1 || flag3)
//<4> Y = (flag1*v1 + flag2*v2 + flag3*v3) * x2

__global__ void linear_bound_switch_mul2D_kernel_4(
	float alpha, const float* __restrict__ X1, float vmin, float vmax,
	float* __restrict__ X2, float v1, float v2, float v3,
	float* __restrict__ Y,
	int lengthv, int width, int stride)
{
	const int step = gridDim.x*blockDim.x, step4 = step << 2;
	const int index = blockIdx.x*blockDim.x + threadIdx.x;

	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		float4 x1 = *(float4*)(X1 + index4);
		float4 x2 = *(float4*)(X2 + index4);

		char4 flag1;//<2> flag3 = (alpha * X1) <= vmin
		flag1.x = (alpha * x1.x) <= vmin;
		flag1.y = (alpha * x1.y) <= vmin;
		flag1.z = (alpha * x1.z) <= vmin;
		flag1.w = (alpha * x1.w) <= vmin;

		char4 flag3;//<1> flag1 = (alpha * X1) >= vmax
		flag3.x = (alpha * x1.x) >= vmax;
		flag3.y = (alpha * x1.y) >= vmax;
		flag3.z = (alpha * x1.z) >= vmax;
		flag3.w = (alpha * x1.w) >= vmax;

		char4 flag2;//flag3 = { vmin < (alpha * X1) < vmax }  = !(flag1 || flag3)
		flag2.x = !(flag1.x || flag3.x);
		flag2.y = !(flag1.y || flag3.y);
		flag2.z = !(flag1.z || flag3.z);
		flag2.w = !(flag1.w || flag3.w);

		float4 y;//<4> Y = (flag1*v1 + flag2*v2 + flag3*v3) * x2
		y.x = (flag1.x * v1 + flag2.x * v2 + flag3.x * v3) * x2.x;
		y.y = (flag1.y * v1 + flag2.y * v2 + flag3.y * v3) * x2.y;
		y.z = (flag1.z * v1 + flag2.z * v2 + flag3.z * v3) * x2.z;
		y.w = (flag1.w * v1 + flag2.w * v2 + flag3.w * v3) * x2.w;

		within_width(y, index4, stride, width);
		*(float4*)(Y + index4) = y;
	}
}

#endif


void __linear_bound_switch_mul2D(cudaStream_t stream,
	float alpha, const float* X1, float vmin, float vmax,
	float* X2, float v1, float v2, float v3,
	float* Y,
	int lengthv, int width, int stride)
{
	if (lengthv < 256) { linear_bound_switch_mul2d_k4_small(stream, alpha, X1, vmin, vmax, X2, v1, v2, v3, Y, lengthv, width, stride); return; }
	if (lengthv > LENGTHV_MAX) { linear_bound_switch_mul2d_k4_max(stream, alpha, X1, vmin, vmax, X2, v1, v2, v3, Y, lengthv, width, stride); return; }
	linear_bound_switch_mul2d_k4(stream, 5, 2, alpha, X1, vmin, vmax, X2, v1, v2, v3, Y, lengthv, width, stride);
}

#endif