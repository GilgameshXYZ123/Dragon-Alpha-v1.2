#pragma once

#ifndef ADAMOD_2D_DECAY_H
#define ADAMOD_2D_DECAY_H

//lengthv = height * stride
//stride = (width + 3)/4*4
//[lenghtv, stride] % 4 == 0
#ifndef ADAMOD_2D_DECAY_CALL
#define ADAMOD_2D_DECAY_CALL

//LB = log2(BLOCK_SIZE)
//LT = log2(THREAD_ELEMENT_VISIT_SIZE)

//lengthv < 256
#define adamod2d_decay_k4_small(stream, W, V, a1, a2, S, b1, b2, eps_t, G, c1, c2, deltaW, lr_t, L1coef, L2coef, lengthv, width, stride)\
	adamod2D_decay_kernel_4\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(W, V,a1,a2, S,b1,b2,eps_t, G,c1,c2, deltaW,lr_t, L1coef, L2coef, lengthv,width,stride)

//common
#define adamod2d_decay_k4(stream, LB, LT, W, V, a1, a2, S, b1, b2, eps_t, G, c1, c2, deltaW, lr_t, L1coef, L2coef, lengthv, width, stride)\
	adamod2D_decay_kernel_4\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(W, V,a1,a2, S,b1,b2,eps_t, G,c1,c2, deltaW,lr_t, L1coef, L2coef, lengthv,width,stride)

//lengthv > lengthv_max
#define adamod2d_decay_k4_max(stream, W, V, a1, a2, S, b1, b2, eps_t, G, c1, c2, deltaW, lr_t, L1coef, L2coef, lengthv, width, stride)\
	adamod2D_decay_kernel_4\
		<<< GRID_MAX, 32, 0, stream >>>\
			(W, V,a1,a2, S,b1,b2,eps_t, G,c1,c2, deltaW,lr_t, L1coef, L2coef, lengthv,width,stride)

#endif


#ifndef ADAMOD_2D_DECAY_KERNEL
#define ADAMOD_2D_DECAY_KERNEL

//Adamod: 3 instrument variables: V, S, G
//<1> V = a1*V + a2*deltaW		# a1 + a2 = 1
//<2> S = b1*S + b2*deltaW^2	# b1 + b2 = 1
//<3> V_correct = S / (1 - Uv)
//<4> S_correct = S / (1 - Us)
//<5> neta = lr / (sqrt(S_correct) + eps)    #adaptive gradient
//<6> G = c1*G + c2*neta		# c1 + c2 = 1
//<7> neta_correct = min(neta, G)
//<8> W = W - neta_correct * V_correct
//
//improved: neta = lr_t /(sqrt(S) + eps_t)
//(1) lr_t = lr * sqrt(1 - Us) / (1 - Uv)
//(2) V = a1*V + a2*deltaW
//(3) S = b1*S + b2*deltaW^2
//(4) V_correct = V / (1 - Uv)
//(5) neta = lr_t / (sqrt(S) + eps)
//(6) G = c1*G + c2*neta
//(7) neta_correct = min(neta, G)
//(8) W = W - neta_correct * V_correct

__global__ void adamod2D_decay_kernel_4(
          float* __restrict__ W,
	      float* __restrict__ V, float a1, float a2, 
	      float* __restrict__ S, float b1, float b2, float eps_t,
	      float* __restrict__ G, float c1, float c2,
	const float* __restrict__ deltaW, float lr_t,
	float L1coef, float L2coef,
	int lengthv, int width, int stride)
{
	const int step = gridDim.x*blockDim.x, step4 = step << 2;
	const int index = blockIdx.x*blockDim.x + threadIdx.x;

	float4 table[2]; table[0] = float4{ 0, 0, 0, 0 };
	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		//load data-------------------------------------------------
		float4 dw = *(float4*)(deltaW + index4);
		float4 v = *(float4*)(V + index4);
		float4 s = *(float4*)(S + index4);
		float4 g = *(float4*)(G + index4);
		float4 w = *(float4*)(W + index4);

		//compute result--------------------------------------------
		//L2: W^2 -> L2coef * W, L1: W   -> L1coef * sign(W)
		dw.x += L1coef * SIGN(w.x) + L2coef * w.x;
		dw.y += L1coef * SIGN(w.y) + L2coef * w.y;
		dw.z += L1coef * SIGN(w.z) + L2coef * w.z;
		dw.w += L1coef * SIGN(w.w) + L2coef * w.w;

		//<1> V = a1*V + a2*deltaW
		v.x = a1 * v.x + a2 * dw.x;
		v.y = a1 * v.y + a2 * dw.y;
		v.z = a1 * v.z + a2 * dw.z;
		v.w = a1 * v.w + a2 * dw.w;
		
		//<2> S = b1*S + b2*deltaW^2
		s.x = b1 * s.x + b2 * (dw.x * dw.x);
		s.y = b1 * s.y + b2 * (dw.y * dw.y);
		s.z = b1 * s.z + b2 * (dw.z * dw.z);
		s.w = b1 * s.w + b2 * (dw.w * dw.w);

		float4 neta;//<3> neta = lr_t / (sqrt(S) + eps_t)
		neta.x = lr_t / (sqrtf(s.x) + eps_t);
		neta.y = lr_t / (sqrtf(s.y) + eps_t);
		neta.z = lr_t / (sqrtf(s.z) + eps_t);
		neta.w = lr_t / (sqrtf(s.w) + eps_t);

		//<4> G = c1*G + c2*neta
		g.x = c1 * g.x + c2 * neta.x;
		g.y = c1 * g.y + c2 * neta.y;
		g.z = c1 * g.z + c2 * neta.z;
		g.w = c1 * g.w + c2 * neta.w;
		
		//W -= min(neta, G) * V
		w.x -= fminf(neta.x, g.x) * v.x;
		w.y -= fminf(neta.y, g.y) * v.y;
		w.z -= fminf(neta.z, g.z) * v.z;
		w.w -= fminf(neta.w, g.w) * v.w;

		//write data------------------------------------------------
		within_width(v, index4, stride, width);
		within_width(s, index4, stride, width);
		within_width_zero_nan(g, index4, table, stride, width);
		within_width_zero_nan(w, index4, table, stride, width);

		*(float4*)(V + index4) = v;//update the velocity
		*(float4*)(S + index4) = s;//update the standard deviation
		*(float4*)(G + index4) = g;//update the step size
		*(float4*)(W + index4) = w;//update weight
	}
}

#endif


void __adamod2D_decay(cudaStream_t stream,
	      float* W,
	      float* V, float a1, float a2,
	      float* S, float b1, float b2, float eps_t,
	      float* G, float c1, float c2,
	const float* deltaW, float lr_t,
	float L1coef, float L2coef,
	int lengthv, int width, int stride)
{
	if (lengthv < 256) { adamod2d_decay_k4_small(stream, W, V, a1, a2, S, b1, b2, eps_t, G, c1, c2, deltaW, lr_t, L1coef, L2coef, lengthv, width, stride); return; }
	if (lengthv > LENGTHV_MAX) { adamod2d_decay_k4_max(stream, W, V, a1, a2, S, b1, b2, eps_t, G, c1, c2, deltaW, lr_t, L1coef, L2coef, lengthv, width, stride); return; }
	adamod2d_decay_k4(stream, 5, 2, W, V, a1, a2, S, b1, b2, eps_t, G, c1, c2, deltaW, lr_t, L1coef, L2coef, lengthv, width, stride);
}

#endif