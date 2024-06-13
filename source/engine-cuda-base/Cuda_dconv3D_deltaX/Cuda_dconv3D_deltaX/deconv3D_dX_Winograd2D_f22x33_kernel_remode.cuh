#pragma once

#ifndef DECONV_3D_DX_WINOGRAD_2D_F22X33_KERNEL_REMODE_V2_H
#define DECONV_3D_DX_WINOGRAD_2D_F22X33_KERNEL_REMODE_V2_H

//Winograd2D Filter transformation:
//convolution:   W[OC, 3, 3, IC] -> G[IC, 4, 4, OC]
//deconvolution: W[OC, 3, 3, IC] -> G[OC, 4, 4, IC]
//(OC, IC) % 4 == 0
//rotate 180:
//[w00, w01, w02]    [w22, w21, w20]
//[w10, w11, w12] -> [w12, w11, w10]
//[w20, w21, w22]    [w02, w01, w00]
#ifndef DECONV_3D_DX_WINOGRAD_2D_KERNEL_REMODE_V2_CALL
#define DECONV_3D_DX_WINOGRAD_2D_KERNEL_REMODE_V2_CALL

//LB = log2(BLOCK_SIZE)

//lengthv = OC * IC
#define deconv3D_dX_winograd2d_f22x33_kremode_v2_k4_small(stream, W, G, lengthv, IC)\
	deconv3D_dX_winograd2D_f22x33_kernel_remode_v2_kernel_4\
		<<< 1, ((lengthv + 3)>>2), 0, stream >>>\
			(W, G, lengthv, IC)

#define deconv3D_dX_winograd2d_f22x33_kremode_v2_k4(stream, LB, LT, W, G, lengthv, IC)\
	deconv3D_dX_winograd2D_f22x33_kernel_remode_v2_kernel_4\
		<<< (lengthv>>LB>>LT), (1<<LB), 0, stream >>>\
			(W, G, lengthv, IC)

#endif


#ifndef DECONV_3D_WINOGRAD_2D_F22X33_KERNEL_REMODE_V2_KERNEL_4
#define DECONV_3D_WINOGRAD_2D_F22X33_KERNEL_REMODE_V2_KERNEL_4

//deconvolution: W[OC, 3, 3, IC] -> CW[OC, 4, 4, IC]
//lengthv = IC * OC, length % 4 == 0

__global__ void deconv3D_dX_winograd2D_f22x33_kernel_remode_v2_kernel_4(
	const float* __restrict__ W,//FH = FW = 3
	      float* __restrict__ G,//FH = FW = 4
	int lengthv, int IC)
{
	const int step = gridDim.x*blockDim.x, step4 = step << 2;
	const int index = blockIdx.x*blockDim.x + threadIdx.x;

	const int IC9 = IC * 9;
	for (int index4 = index << 2; index4 < lengthv; index4 += step4)//[oc, ic0]
	{
		//======[load 4 * (3 * 3) elements from W]=======================================================
		const int oc = index4 / IC, ic = index4 - oc * IC;
		const int W00 = oc * IC9 + ic, W01 = W00 + IC, W02 = W01 + IC;//[oc, 0, 0-2, ic0-ic3]
		const int W10 = W00 + IC * 3 , W11 = W10 + IC, W12 = W11 + IC;//[oc, 1, 0-2, ic0-ic3]  
		const int W20 = W10 + IC * 3 , W21 = W20 + IC, W22 = W21 + IC;//[oc, 2, 0-2, ic0-ic3]

		//[w00, w01, w02]    [w22, w21, w20]
		//[w10, w11, w12] -> [w12, w11, w10]
		//[w20, w21, w22]    [w02, w01, w00]
		float4 w22 = *(float4*)(W + W00), w21 = *(float4*)(W + W01), w20 = *(float4*)(W + W02);
		float4 w12 = *(float4*)(W + W10), w11 = *(float4*)(W + W11), w10 = *(float4*)(W + W12);
		float4 w02 = *(float4*)(W + W20), w01 = *(float4*)(W + W21), w00 = *(float4*)(W + W22);

		//======[transform W(3*3) -> G(4*4) ]============================================================
		float4 b00, b10, b20, b30;
		float4 b01, b11, b21, b31;
		float4 b02, b12, b22, b32; {
			b00.x = w00.x; b10.x = 0.5f * (w00.x + w10.x + w20.x); b20.x = 0.5f * (w00.x - w10.x + w20.x); b30.x = w20.x;//ic0
			b01.x = w01.x; b11.x = 0.5f * (w01.x + w11.x + w21.x); b21.x = 0.5f * (w01.x - w11.x + w21.x); b31.x = w21.x;
			b02.x = w02.x; b12.x = 0.5f * (w02.x + w12.x + w22.x); b22.x = 0.5f * (w02.x - w12.x + w22.x); b32.x = w22.x;

			b00.y = w00.y; b10.y = 0.5f * (w00.y + w10.y + w20.y); b20.y = 0.5f * (w00.y - w10.y + w20.y); b30.y = w20.y;//ic1
			b01.y = w01.y; b11.y = 0.5f * (w01.y + w11.y + w21.y); b21.y = 0.5f * (w01.y - w11.y + w21.y); b31.y = w21.y;
			b02.y = w02.y; b12.y = 0.5f * (w02.y + w12.y + w22.y); b22.y = 0.5f * (w02.y - w12.y + w22.y); b32.y = w22.y;

			b00.z = w00.z; b10.z = 0.5f * (w00.z + w10.z + w20.z); b20.z = 0.5f * (w00.z - w10.z + w20.z); b30.z = w20.z;//ic2
			b01.z = w01.z; b11.z = 0.5f * (w01.z + w11.z + w21.z); b21.z = 0.5f * (w01.z - w11.z + w21.z); b31.z = w21.z;
			b02.z = w02.z; b12.z = 0.5f * (w02.z + w12.z + w22.z); b22.z = 0.5f * (w02.z - w12.z + w22.z); b32.z = w22.z;

			b00.w = w00.w; b10.w = 0.5f * (w00.w + w10.w + w20.w); b20.w = 0.5f * (w00.w - w10.w + w20.w); b30.w = w20.w;//ic3
			b01.w = w01.w; b11.w = 0.5f * (w01.w + w11.w + w21.w); b21.w = 0.5f * (w01.w - w11.w + w21.w); b31.w = w21.w;
			b02.w = w02.w; b12.w = 0.5f * (w02.w + w12.w + w22.w); b22.w = 0.5f * (w02.w - w12.w + w22.w); b32.w = w22.w;
		}
	
		float4 g00, g01, g02, g03;
		float4 g10, g11, g12, g13;
		float4 g20, g21, g22, g23;
		float4 g30, g31, g32, g33; {
			g00.x = b00.x; g01.x = 0.5f * (b00.x + b01.x + b02.x), g02.x = 0.5f * (b00.x - b01.x + b02.x), g03.x = b02.x;
			g10.x = b10.x; g11.x = 0.5f * (b10.x + b11.x + b12.x), g12.x = 0.5f * (b10.x - b11.x + b12.x), g13.x = b12.x;
			g20.x = b20.x, g21.x = 0.5f * (b20.x + b21.x + b22.x), g22.x = 0.5f * (b20.x - b21.x + b22.x), g23.x = b22.x;
			g30.x = b30.x, g31.x = 0.5f * (b30.x + b31.x + b32.x), g32.x = 0.5f * (b30.x - b31.x + b32.x), g33.x = b32.x;

			g00.y = b00.y; g01.y = 0.5f * (b00.y + b01.y + b02.y), g02.y = 0.5f * (b00.y - b01.y + b02.y), g03.y = b02.y;
			g10.y = b10.y; g11.y = 0.5f * (b10.y + b11.y + b12.y), g12.y = 0.5f * (b10.y - b11.y + b12.y), g13.y = b12.y;
			g20.y = b20.y, g21.y = 0.5f * (b20.y + b21.y + b22.y), g22.y = 0.5f * (b20.y - b21.y + b22.y), g23.y = b22.y;
			g30.y = b30.y, g31.y = 0.5f * (b30.y + b31.y + b32.y), g32.y = 0.5f * (b30.y - b31.y + b32.y), g33.y = b32.y;

			g00.z = b00.z; g01.z = 0.5f * (b00.z + b01.z + b02.z), g02.z = 0.5f * (b00.z - b01.z + b02.z), g03.z = b02.z;
			g10.z = b10.z; g11.z = 0.5f * (b10.z + b11.z + b12.z), g12.z = 0.5f * (b10.z - b11.z + b12.z), g13.z = b12.z;
			g20.z = b20.z, g21.z = 0.5f * (b20.z + b21.z + b22.z), g22.z = 0.5f * (b20.z - b21.z + b22.z), g23.z = b22.z;
			g30.z = b30.z, g31.z = 0.5f * (b30.z + b31.z + b32.z), g32.z = 0.5f * (b30.z - b31.z + b32.z), g33.z = b32.z;

			g00.w = b00.w; g01.w = 0.5f * (b00.w + b01.w + b02.w), g02.w = 0.5f * (b00.w - b01.w + b02.w), g03.w = b02.w;
			g10.w = b10.w; g11.w = 0.5f * (b10.w + b11.w + b12.w), g12.w = 0.5f * (b10.w - b11.w + b12.w), g13.w = b12.w;
			g20.w = b20.w, g21.w = 0.5f * (b20.w + b21.w + b22.w), g22.w = 0.5f * (b20.w - b21.w + b22.w), g23.w = b22.w;
			g30.w = b30.w, g31.w = 0.5f * (b30.w + b31.w + b32.w), g32.w = 0.5f * (b30.w - b31.w + b32.w), g33.w = b32.w;
		}

		//======[load 4 * (4 * 4) elements to G]=========================================================
		const int G00 = (oc << 4)*IC + ic, G01 = G00 + IC, G02 = G00 + (IC << 1), G03 = G00 + IC * 3;
		const int G10 = G00 + (IC << 2)  , G11 = G10 + IC, G12 = G10 + (IC << 1), G13 = G10 + IC * 3;
		const int G20 = G10 + (IC << 2)  , G21 = G20 + IC, G22 = G20 + (IC << 1), G23 = G20 + IC * 3;
		const int G30 = G20 + (IC << 2)  , G31 = G30 + IC, G32 = G30 + (IC << 1), G33 = G30 + IC * 3;

		*(float4*)(G + G00) = g00; *(float4*)(G + G01) = g01; *(float4*)(G + G02) = g02; *(float4*)(G + G03) = g03;
		*(float4*)(G + G10) = g10; *(float4*)(G + G11) = g11; *(float4*)(G + G12) = g12; *(float4*)(G + G13) = g13;
		*(float4*)(G + G20) = g20; *(float4*)(G + G21) = g21; *(float4*)(G + G22) = g22; *(float4*)(G + G23) = g23;
		*(float4*)(G + G30) = g30; *(float4*)(G + G31) = g31; *(float4*)(G + G32) = g32; *(float4*)(G + G33) = g33;
	}
}

#endif


//OC % 4 == 0, IC % 4 == 0
void __deconv3D_dX_winograd2D_f22x33_kernel_remode_v2(cudaStream_t stream,
	const float* W, //[OC, 3, 3, IC]
	      float *G, //[OC, 4, 4, IC]
	int OC, int IC)
{
	int lengthv = OC * IC;
	if (lengthv < 256) { deconv3D_dX_winograd2d_f22x33_kremode_v2_k4_small(stream, W, G, lengthv, IC); return; }//small
	deconv3D_dX_winograd2d_f22x33_kremode_v2_k4(stream, 5, 2, W, G, lengthv, IC); return;
}

#endif
