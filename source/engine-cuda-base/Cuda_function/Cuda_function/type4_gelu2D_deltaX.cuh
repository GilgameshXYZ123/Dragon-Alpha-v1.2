#pragma once


#ifndef GELU_2D_DELTAX_H
#define FELU_2D_DELTAX_H

//lengthv = height * stride
//stride = (width + 3)/4*4
//[lenghtv, stride] % 4 == 0
#ifndef GELU_2D_DELTAX_CALL
#define FELU_2D_DELTAX_CALL

//LB = log2(BLOCK_SIZE)
//LT = log2(THREAD_ELEMENT_VISIT_SIZE)

//lengthv < 256
#define gelu2d_deltaX_k4_small(stream,  deltaX, deltaY, X, lengthv, width, stride)\
	gelu2D_deltaX_kernel_4\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(deltaX, deltaY, X, lengthv, width, stride)

//common
#define gelu2d_deltaX_k4(stream, LB, LT, deltaX, deltaY, X, lengthv, width, stride)\
	gelu2D_deltaX_kernel_4\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(deltaX, deltaY, X, lengthv, width, stride)

//lengthv > lengthv_max
#define gelu2d_deltaX_k4_max(stream,  deltaX, deltaY, X, lengthv, width, stride)\
	gelu2D_deltaX_kernel_4\
		<<< GRID_MAX, 32, 0, stream >>>\
			(deltaX, deltaY, X, lengthv, width, stride)

#endif


#ifndef GELU_2D_DELTAX_KERNEL
#define FELU_2D_DELTAX_KERNEL

//[Forward]
//(1) a = -1.5957692, b = 0.044715
//(2) u = a * x * (1 + b * x^2)
//(3) y = x / (1 + e^u)
//
//[Backward]
//(1) B = 1 + e^u
//(2) A = 1 - (e^u / B) * (u + 2ab*x^2)
//(3) y' = A / B
//STEP:
//(1) u = -1.5957692 * x * (1 + 0.044715 * x^2)
//(2) expu = exp(u)
//(3) B = 1 / (1 + expu)
//(4) A = 1 - (expu * B) * (u - 0.14270963 * x^2)
//(5) deltaX = deltaY * (A * B)

__global__ void gelu2D_deltaX_kernel_4(
	      float* __restrict__ deltaX,
	const float* __restrict__ deltaY,
	const float* __restrict__ X,
	int lengthv, int width, int stride)
{
	const int step = gridDim.x*blockDim.x, step4 = step << 2;
	const int index = blockIdx.x*blockDim.x + threadIdx.x;

	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		float4 dy = *(float4*)(deltaY + index4);
		float4 x = *(float4*)(X + index4);

		float4 u;//u = -1.5957692 * x * (1 + 0.044715 * x^2)
		u.x = -1.5957692f * x.x * (1.0f + 0.044715f * x.x * x.x);
		u.y = -1.5957692f * x.y * (1.0f + 0.044715f * x.y * x.y);
		u.z = -1.5957692f * x.z * (1.0f + 0.044715f * x.z * x.z);
		u.w = -1.5957692f * x.w * (1.0f + 0.044715f * x.w * x.w);

		//====[gate1: x > 0, u < 0, e^u != inf]=================================
		float4 expu, A1, B1;//A1 = 1 / (1 + e^u), B1 = e^u / (1 + e^u)
		expu.x = expf(u.x); A1.x = 1.0f / (expu.x + 1.0f); B1.x = expu.x * A1.x;
		expu.y = expf(u.y); A1.y = 1.0f / (expu.y + 1.0f); B1.y = expu.y * A1.y;
		expu.z = expf(u.z); A1.z = 1.0f / (expu.z + 1.0f); B1.z = expu.z * A1.z;
		expu.w = expf(u.w); A1.w = 1.0f / (expu.w + 1.0f); B1.w = expu.w * A1.w;

		//====[gate2: x < 0, u > 0, e^-u != inf]================================
		float4 expm, A2, B2;//A2 = e^-u / (1 + e^-u), B2 = 1 / (1 + e^-u)
		expm.x = expf(-u.x); B2.x = 1.0f / (expm.x + 1.0f); A2.x = expm.x * B2.x;
		expm.y = expf(-u.y); B2.y = 1.0f / (expm.y + 1.0f); A2.y = expm.y * B2.y;
		expm.z = expf(-u.z); B2.z = 1.0f / (expm.z + 1.0f); A2.z = expm.z * B2.z;
		expm.w = expf(-u.w); B2.w = 1.0f / (expm.w + 1.0f); A2.w = expm.w * B2.w;

		//======[Switch]======================================================
		float4 As[2] = { A2, A1 }, Bs[2] = { B2, B1 };
		char4 flag; float4 A, B;
		flag.x = (x.x > 0.0f); A.x = As[flag.x].x; B.x = Bs[flag.x].x;
		flag.y = (x.y > 0.0f); A.y = As[flag.y].y; B.y = Bs[flag.y].y;
		flag.z = (x.z > 0.0f); A.z = As[flag.z].z; B.z = Bs[flag.z].z;
		flag.w = (x.w > 0.0f); A.w = As[flag.w].w; B.w = Bs[flag.w].w;

		float4 dx;//dx = A * (1 - B * (u - 0.14270963f * x^3))
		dx.x = A.x * (1.0f - B.x * (u.x - 0.14270963f * x.x * x.x * x.x));
		dx.y = A.y * (1.0f - B.y * (u.y - 0.14270963f * x.y * x.y * x.y));
		dx.z = A.z * (1.0f - B.z * (u.z - 0.14270963f * x.z * x.z * x.z));
		dx.w = A.w * (1.0f - B.w * (u.w - 0.14270963f * x.w * x.w * x.w));

		simdMul4(dx, dx, dy);//deltaX = deltaY * driY
		within_width(dx, index4, stride, width);
		*(float4*)(deltaX + index4) = dx;
	}
}

#endif


void __gelu2D_deltaX(cudaStream_t stream,
	float* deltaX,
	const float* deltaY,
	const float* X,
	int lengthv, int width, int stride)
{
	if (lengthv < 256) { gelu2d_deltaX_k4_small(stream, deltaX, deltaY, X, lengthv, width, stride); return; }
	if (lengthv > LENGTHV_MAX) { gelu2d_deltaX_k4_max(stream, deltaX, deltaY, X, lengthv, width, stride); return; }
	gelu2d_deltaX_k4(stream, 5, 2, deltaX, deltaY, X, lengthv, width, stride);
}

#endif