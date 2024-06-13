#pragma once

#ifndef RMSPROP_2D_H
#define RMSPROP_2D_H

//lengthv = height * stride
//stride = (width + 3)/4*4
//[lenghtv, stride] % 4 == 0
#ifndef RMSPROP_2D_CALL
#define RMSPROP_2D_CALL

//LB = log2(BLOCK_SIZE)
//LT = log2(THREAD_ELEMENT_VISIT_SIZE)

//lengthv < 256
#define rmsprop2d_k4_small(stream, W, S, a1, a2, eps_t, deltaW, lr_t, lengthv, width, stride)\
	rmsprop2D_kernel_4\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(W, S,a1,a2,eps_t, deltaW,lr_t, lengthv,width,stride)

//common
#define rmsprop2d_k4(stream, LB, LT, W, S, a1, a2, eps_t, deltaW, lr_t, lengthv, width, stride)\
	rmsprop2D_kernel_4\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(W, S,a1,a2,eps_t, deltaW,lr_t, lengthv,width,stride)

//lengthv > lengthv_max
#define rmsprop2d_k4_max(stream, W, S, a1, a2, eps_t, deltaW, lr_t, lengthv, width, stride)\
	rmsprop2D_kernel_4\
		<<< GRID_MAX, 32, 0, stream >>>\
			(W, S,a1,a2,eps_t, deltaW,lr_t, lengthv,width,stride)

#endif


#ifndef RMSPROP_2D_KERNEL
#define RMSPROP_2D_KERNEL

//<1> S = a1*S + a2*deltaW^2
//<2> W = W - lr_t * deltaW / (sqrt(S) + eps_t)

__global__ void rmsprop2D_kernel_4(
	      float* __restrict__ W,
	      float* __restrict__ S, float a1, float a2, float eps_t,
	const float* __restrict__ deltaW, float lr_t,
	int lengthv, int width, int stride)
{
	const int step = gridDim.x*blockDim.x, step4 = step << 2;
	const int index = blockIdx.x*blockDim.x + threadIdx.x;

	float4 table[2]; table[0] = float4{ 0, 0, 0, 0 };
	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		//load data-------------------------------------------------
		float4 dw = *(float4*)(deltaW + index4);
		float4 s = *(float4*)(S + index4);
		float4 w = *(float4*)(W + index4);
		
		//compute result--------------------------------------------
		//<1> S = a1*S + a2*deltaW^2
		s.x = a1 * s.x + a2 * (dw.x * dw.x);
		s.y = a1 * s.y + a2 * (dw.y * dw.y);
		s.z = a1 * s.z + a2 * (dw.z * dw.z);
		s.w = a1 * s.w + a2 * (dw.w * dw.w);

		float4 step;
		step.x = dw.x / (sqrtf(s.x) + eps_t);
		step.y = dw.y / (sqrtf(s.y) + eps_t);
		step.z = dw.z / (sqrtf(s.z) + eps_t);
		step.w = dw.w / (sqrtf(s.w) + eps_t);

		//<2> W = W - lr_t * deltaW / (sqrt(S) + eps_t)   
		w.x -= lr_t * step.x;
		w.y -= lr_t * step.y;
		w.z -= lr_t * step.z;
		w.w -= lr_t * step.w;

		//write data------------------------------------------------
		within_width(s, index4, stride, width);
		within_width_zero_nan(w, index4, table, stride, width);

		*(float4*)(S + index4) = s;//update the standard deviation
		*(float4*)(W + index4) = w;//update weight
	}
}

#endif


void __rmsprop2D(cudaStream_t stream,
	      float* W,
	      float* S, float a1, float a2, float eps_t,
	const float* deltaW, float lr_t,
	int lengthv, int width, int stride)
{
	if (lengthv < 256) { rmsprop2d_k4_small(stream, W, S, a1, a2, eps_t, deltaW, lr_t, lengthv, width, stride); return; }
	if (lengthv > LENGTHV_MAX) { rmsprop2d_k4_max(stream, W, S, a1, a2, eps_t, deltaW, lr_t, lengthv, width, stride); return; }
	rmsprop2d_k4(stream, 5, 2, W, S, a1, a2, eps_t, deltaW, lr_t, lengthv, width, stride);
}

#endif