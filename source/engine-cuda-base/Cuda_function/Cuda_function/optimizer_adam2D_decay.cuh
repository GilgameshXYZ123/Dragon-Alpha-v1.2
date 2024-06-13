#pragma once

#ifndef ADAM_2D_DECAY_H
#define ADAM_2D_DECAY_H

//lengthv = height * stride
//stride = (width + 3)/4*4
//[lenghtv, stride] % 4 == 0
#ifndef ADAM_2D_DECAY_CALL
#define ADAM_2D_DECAY_CALL

//LB = log2(BLOCK_SIZE)
//LT = log2(THREAD_ELEMENT_VISIT_SIZE)

//lengthv < 256
#define adam2d_decay_k4_small(stream, W, V, a1, a2, S, b1, b2, eps_t, deltaW, lr_t, L1coef, L2coef, lengthv, width, stride)\
	adam2D_decay_kernel_4\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(W, V,a1,a2, S,b1,b2,eps_t, deltaW,lr_t, L1coef, L2coef, lengthv,width,stride)

//common
#define adam2d_decay_k4(stream, LB, LT, W, V, a1, a2, S, b1, b2, eps_t, deltaW, lr_t, L1coef, L2coef, lengthv, width, stride)\
	adam2D_decay_kernel_4\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(W, V,a1,a2, S,b1,b2,eps_t, deltaW,lr_t, L1coef, L2coef, lengthv,width,stride)

//lengthv > lengthv_max
#define adam2d_decay_k4_max(stream, W, V, a1, a2, S, b1, b2, eps_t, deltaW, lr_t, L1coef, L2coef, lengthv, width, stride)\
	adam2D_decay_kernel_4\
		<<< GRID_MAX, 32, 0, stream >>>\
			(W, V,a1,a2, S,b1,b2,eps_t, deltaW,lr_t, L1coef, L2coef, lengthv,width,stride)

#endif


#ifndef ADAM_2D_DECAY_KERNEL
#define ADAM_2D_DECAY_KERNEL

//<1> V = a1 * V + a2 * deltaW        
//<2> S = b1 * S + b2 * deltaW ^ 2
//<3> W = W - lr_t * V / (sqrt(S) + eps_t)

__global__ void adam2D_decay_kernel_4(
          float* __restrict__ W,
	      float* __restrict__ V, float a1, float a2,
	      float* __restrict__ S, float b1, float b2, float eps_t,
	const float* __restrict__ deltaW, float lr_t,
	float L1coef, float L2coef,
	int lengthv, int width, int stride)
{
	const int step = gridDim.x*blockDim.x, step4 = step << 2;
	const int index = blockIdx.x*blockDim.x + threadIdx.x;

	float4 table[2]; table[0] = F32_4_0;
	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		//load data----------------------------------------------
		float4 dw = *(float4*)(deltaW + index4);
		float4 v = *(float4*)(V + index4);
		float4 s = *(float4*)(S + index4);
		float4 w = *(float4*)(W + index4);

		//compute result-----------------------------------------
		//L2: W^2 -> L2coef * W, L1: W -> L1coef * sign(W)
		dw.x += L1coef * SIGN(w.x) + L2coef * w.x;
		dw.y += L1coef * SIGN(w.y) + L2coef * w.y;
		dw.z += L1coef * SIGN(w.z) + L2coef * w.z;
		dw.w += L1coef * SIGN(w.w) + L2coef * w.w;

		//V = a1*V + a2*deltaW  
		v.x = a1 * v.x + a2 * dw.x;
		v.y = a1 * v.y + a2 * dw.y;
		v.z = a1 * v.z + a2 * dw.z;
		v.w = a1 * v.w + a2 * dw.w;
		
		//S = b1*S + b2*deltaW^2
		s.x = b1 * s.x + b2 * (dw.x * dw.x);
		s.y = b1 * s.y + b2 * (dw.y * dw.y);
		s.z = b1 * s.z + b2 * (dw.z * dw.z);
		s.w = b1 * s.w + b2 * (dw.w * dw.w);

		//W = W - lr_t * V / (sqrt(S) + eps_t)
		w.x -= lr_t * (v.x / (sqrtf(s.x) + eps_t));
		w.y -= lr_t * (v.y / (sqrtf(s.y) + eps_t));
		w.z -= lr_t * (v.z / (sqrtf(s.z) + eps_t));
		w.w -= lr_t * (v.w / (sqrtf(s.w) + eps_t));

		//write data---------------------------------------------
		within_width(v, index4, stride, width);
		within_width(s, index4, stride, width);
		within_width_zero_nan(w, index4, table, stride, width);

		*(float4*)(V + index4) = v;//update the velocity
		*(float4*)(S + index4) = s;//update the standard deviation
		*(float4*)(W + index4) = w;//update weight
	}
}

#endif


void __adam2D_decay(cudaStream_t stream,
	      float* W,
	      float* V, float a1, float a2,
	      float* S, float b1, float b2, float eps,
	const float* deltaW, float lr_t,
	float L1coef, float L2coef,
	int lengthv, int width, int stride)
{
	if (lengthv < 256) { adam2d_decay_k4_small(stream, W, V, a1, a2, S, b1, b2, eps, deltaW, lr_t, L1coef, L2coef, lengthv, width, stride); return; }
	if (lengthv > LENGTHV_MAX) { adam2d_decay_k4_max(stream, W, V, a1, a2, S, b1, b2, eps, deltaW, lr_t, L1coef, L2coef, lengthv, width, stride); return; }
	adam2d_decay_k4(stream, 5, 2, W, V, a1, a2, S, b1, b2, eps, deltaW, lr_t, L1coef, L2coef, lengthv, width, stride);
}

#endif