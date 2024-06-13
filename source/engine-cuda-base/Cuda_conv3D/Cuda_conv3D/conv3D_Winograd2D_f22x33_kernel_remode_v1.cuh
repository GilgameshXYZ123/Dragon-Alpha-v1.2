#pragma once

#ifndef WINOGRAD_2D_F22X33_KERNEL_REMODE_V1_H
#define WINOGRAD_2D_F22X33_KERNEL_REMODE_V1_H

//Winograd2D Filter transformation:
//W[OC, 3, 3, IC] -> CW[OC, IC, 4, 4]
#ifndef WINOGRAD_2D_KERNEL_REMODE_V1_CALL
#define WINOGRAD_2D_KERNEL_REMODE_V1_CALL

//LB = log2(BLOCK_SIZE)
//lengthv = OC * IC

#define winograd2d_f22x33_kremode_k1_small(stream, W, G, lengthv, IC, OC)\
	winograd2D_f22x33_kernel_remode_kernel_1\
		<<< 1, ((lengthv + 3)>>2), 0, stream >>>\
			(W, G, lengthv, IC, OC)

#define winograd2d_f22x33_kremode_k1(stream, LB, LT, W, G, lengthv, IC, OC)\
	winograd2D_f22x33_kernel_remode_kernel_1\
		<<< (lengthv>>LB>>LT), (1<<LB), 0, stream >>>\
			(W, G, lengthv, IC, OC)

#endif


#ifndef WINOGRAD_2D_F22X33_KERNEL_REMODE_KERNEL_1
#define WINOGRAD_2D_F22X33_KERNEL_REMODE_KERNEL_1

//lengthv = OC * IC, IC % 4 == 0
__global__ void winograd2D_f22x33_kernel_remode_kernel_1(
	const float* __restrict__ W,//FH = FW = 3
	      float* __restrict__ G,//FH = FW = 4
	int lengthv, int IC, int OC)
{
	int step = gridDim.x*blockDim.x;
	int index = blockIdx.x*blockDim.x + threadIdx.x;

	const int IC9 = IC * 9;
	for (; index < lengthv; index += step)
	{
		const int oc = index / IC, ic = index % IC;//index = (oc, ic)

		const int W00 = oc * IC9 + ic, W01 = W00 + IC, W02 = W01 + IC;//[oc, 0, 0-2, ic]
		const int W10 = W02 + IC, W11 = W10 + IC, W12 = W11 + IC;     //[oc, 1, 0-2, ic]  
		const int W20 = W12 + IC, W21 = W20 + IC, W22 = W21 + IC;     //[oc, 2, 0-2, ic]

		float w00 = W[W00], w01 = W[W01], w02 = W[W02];
		float w10 = W[W10], w11 = W[W11], w12 = W[W12];
		float w20 = W[W20], w21 = W[W21], w22 = W[W22];

		//transform---------------------------------------------------------------------------------
		float b00 = w00, b10 = 0.5f * (w00 + w10 + w20), b20 = 0.5f * (w00 - w10 + w20), b30 = w20;
		float b01 = w01, b11 = 0.5f * (w01 + w11 + w21), b21 = 0.5f * (w01 - w11 + w21), b31 = w21;
		float b02 = w02, b12 = 0.5f * (w02 + w12 + w22), b22 = 0.5f * (w02 - w12 + w22), b32 = w22;

		float g00 = b00, g01 = 0.5f * (b00 + b01 + b02), g02 = 0.5f * (b00 - b01 + b02), g03 = b02;
		float g10 = b10, g11 = 0.5f * (b10 + b11 + b12), g12 = 0.5f * (b10 - b11 + b12), g13 = b12;
		float g20 = b20, g21 = 0.5f * (b20 + b21 + b22), g22 = 0.5f * (b20 - b21 + b22), g23 = b22;
		float g30 = b30, g31 = 0.5f * (b30 + b31 + b32), g32 = 0.5f * (b30 - b31 + b32), g33 = b32;
		//transform---------------------------------------------------------------------------------

		const int G0 = (ic*OC + oc) << 4;//[ic, oc, 0, 0]
		*(float4*)(G + G0     ) = float4{ g00, g01, g02, g03 };//[oc, ic, 0, 0-3]
		*(float4*)(G + G0 +  4) = float4{ g10, g11, g12, g13 };//[oc, ic, 1, 0-3]
		*(float4*)(G + G0 +  8) = float4{ g20, g21, g22, g23 };//[oc, ic, 2, 0-3]
		*(float4*)(G + G0 + 12) = float4{ g30, g31, g32, g33 };//[oc, ic, 3, 0-3]
	}
}

#endif


void __winograd2D_f22x33_kernel_remode(cudaStream_t stream,
	const float* W, float *G, 
	int OC, int IC) 
{
	int lengthv = OC * IC;
	if (lengthv < 256)  { winograd2d_f22x33_kremode_k1_small(stream, W, G, lengthv, IC, OC); return; }
	if (lengthv < 1024) { winograd2d_f22x33_kremode_k1(stream, 5, 0, W, G, lengthv, IC, OC); return; }
	if (lengthv < 4096) { winograd2d_f22x33_kremode_k1(stream, 5, 1, W, G, lengthv, IC, OC); return; }
	winograd2d_f22x33_kremode_k1(stream, 5, 2, W, G, lengthv, IC, OC); return;
}

#endif