#pragma once

#ifndef CONV_3D_INPUT_PAD_REMORE_H
#define CONV_3D_INPUT_PAD_REMODE_H

//X[(N, IH, IW), IC] -> D[IC, N, DH, DW]
//<1> DH = IH + 2*ph = (OH - 1)*sh + FH 
//<2> DW = IW + 2*pw = (OW - 1)*sw + FW
//(OH, OW) % 2 == 0
#ifndef CONV_3D_INPUT_PAD_REMORE_CALL
#define CONV_3D_INPUT_PAD_REMORE_CALL

//LB = log2(BLOCK_SIZE)
//lengthv = <N * DH * DW> * IC % 4 == 0

//======[basic approach]===========================================
#define input_pad_remode_k4_small(stream, X, IH, IW, D, DH, DW, N, IC, ph, pw, lengthv)\
	input_pad_remode_kernel_4\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(X, IH, IW, D, DH, DW, N, IC, ph, pw, lengthv)

#define input_pad_remode_k4(stream, LB, LT, X, IH, IW, D, DH, DW, N, IC, ph, pw, lengthv)\
	input_pad_remode_kernel_4\
		<<< (lengthv>>LB>>LT), 1<<LB, 0, stream >>>\
			(X, IH, IW, D, DH, DW, N, IC, ph, pw, lengthv)

//======[mat transpose approach]===================================
//LTY>>2, LTX>>2
//GM = N * DH * DW
#define input_pad_remode_k44(stream, LBY, LBX, LTY, LTX, X, IH, IW, D, DH, DW, N, IC, ph, pw, GM)\
	input_pad_remode_kernel_4_4\
		<<< dim3((GM>>LBX>>LTX), (IC>>LBY>>LTY)), dim3(1<<LBX, 1<<LBY), 0, stream >>>\
			(X, IH, IW, D, (DH*DW), DW, GM, IC, ph, pw)

#define input_pad_remode_k44S(stream, LB, LT, X, IH, IW, D, DH, DW, N, IC, ph, pw, GM)\
	input_pad_remode_kernel_4_4S<LB>\
		<<< dim3(MIN_int32(IC>>LB>>LT, GRID_MAX),\
                 MIN_int32(GM>>LB>>LT, GRID_MAX)),\
            dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, D, (DH*DW), DW, GM, IC, ph, pw)

#endif


//======[basic approach]===========================================
#ifndef CONV3D_3D_INPUT_PAD_REMORE_KERNEL_4
#define CONV3D_3D_INPUT_PAD_REMORE_KERNEL_4

//lengthv = <N * DH * DW> * IC
__global__ void input_pad_remode_kernel_4(
	const float* __restrict__ X, int IH, int IW,//[<N, IH, IW>, IC]
	      float* __restrict__ D, int DH, int DW,//[IC, <N, DH, DW>]
	int N, int IC, int ph, int pw, int lengthv)
{
	int step = gridDim.x*blockDim.x, step4 = step << 2;
	int index = blockIdx.x*blockDim.x + threadIdx.x;

	int DW_IC = DW * IC;
	int DH_DW_IC = DH * DW_IC;
	int Dstride = N * DH * DW;

	//index4 = [n, dh, dw, ic]
	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		int n = index4 / DH_DW_IC, ir = index4 - n * DH_DW_IC;
		int dh = ir / DW_IC; ir -= dh * DW_IC;
		int dw = ir / IC, ic = ir - dw * IC;

		//read 4 elements from X
		const int ih = dh - ph, iw = dw - pw;
		bool lx = (ih >= 0) && (iw >= 0) && (ih < IH) && (iw < IW);
		const int xoffset = ((n*IH + ih)*IW + iw)*IC + ic;
		float4 x = (lx ? *(float4*)(X + xoffset) : F32_4_0);

		//write 4 elements to D
		const int D0 = ((ic*N + n)*DH + dh)*DW + dw;//[ic, n, dh, dw]
		const int D1 = D0 + Dstride;               
		const int D2 = D0 + (Dstride << 1);         
		const int D3 = D0 + Dstride * 3;            
		D[D0] = x.x;//[ic0, n, dh, dw]
		D[D1] = x.y;//[ic1, n, dh, dw]
		D[D2] = x.z;//[ic2, n, dh, dw]
		D[D3] = x.w;//[ic3, n, dh, dw]
	}
}

#endif


//======[mat transpose approach]===================================
#ifndef CONV_3D_INPUT_PAD_REMORE_KERNEL_4_4
#define CONV_3D_INPUT_PAD_REMORE_KERNEL_4_4

//[<N, IH, IW>, IC] -> [IC, <N, IH, IW>]
//(by, ty) -> y -> ic
//(bx, tx) -> x -> j

//ph = pw = 1:
//[N, IH, IW, IC] = [ 64, 32, 32, 128]: 304.256 GB/s
//[N, IH, IW, IC] = [256, 16, 16, 128]: 290.191 GB/s
//[N, IH, IW, IC] = [256,  8,  8, 256]: 230.492 GB/s
//[N, IH, IW, IC] = [512,  4,  4, 512]: 157.151 GB/s

__global__ void input_pad_remode_kernel_4_4(
	const float* __restrict__ X, int IH, int IW,
	      float* __restrict__ D, int DH_DW, int DW,
	int GM, int IC, int ph, int pw)
{
	const int y = (blockIdx.y * blockDim.y) + threadIdx.y;//IC / 4
	const int x = (blockIdx.x * blockDim.x) + threadIdx.x;//(N * DN * DW) / 4
	
	const int stepY = blockDim.y * gridDim.y, stepY4 = stepY << 2;
	const int stepX = blockDim.x * gridDim.x, stepX4 = stepX << 2;

	for (int y4 = y << 2; y4 < IC; y4 += stepY4)//y4 -> [ic0 -> ic4]
	for (int x4 = x << 2; x4 < GM; x4 += stepX4)//x4 -> [j0 - j4] : (n, ih, iw)
	{
		int c0 = y4, c1 = c0 + 1, c2 = c0 + 2, c3 = c0 + 3;
		int j0 = x4, j1 = j0 + 1, j2 = j0 + 2, j3 = j0 + 3;
		
		//======[load: 4*4 elements from X]========================================================
		int n0, dh0, dw0; { n0 = j0 / DH_DW; int jr = j0 - n0 * DH_DW; dh0 = jr / DW; dw0 = jr - dh0 * DW; }
		int n1, dh1, dw1; { n1 = j1 / DH_DW; int jr = j1 - n1 * DH_DW; dh1 = jr / DW; dw1 = jr - dh1 * DW; }
		int n2, dh2, dw2; { n2 = j2 / DH_DW; int jr = j2 - n2 * DH_DW; dh2 = jr / DW; dw2 = jr - dh2 * DW; }
		int n3, dh3, dw3; { n3 = j3 / DH_DW; int jr = j3 - n3 * DH_DW; dh3 = jr / DW; dw3 = jr - dh3 * DW; }

		int ih0 = dh0 - ph, iw0 = dw0 - pw;
		int ih1 = dh1 - ph, iw1 = dw1 - pw;
		int ih2 = dh2 - ph, iw2 = dw2 - pw;
		int ih3 = dh3 - ph, iw3 = dw3 - pw;

		int xoffset0 = ((n0*IH + ih0)*IW + iw0)*IC + c0;//[c0 - c3], IC % 4 == 0, c0 % 4 == 0
		int xoffset1 = ((n1*IH + ih1)*IW + iw1)*IC + c0;//[c0 - c3]
		int xoffset2 = ((n2*IH + ih2)*IW + iw2)*IC + c0;//[c0 - c3]
		int xoffset3 = ((n3*IH + ih3)*IW + iw3)*IC + c0;//[c0 - c3]

		bool lx0 = (ih0 >= 0) && (ih0 < IH) && (iw0 >= 0) && (iw0 < IW);
		bool lx1 = (ih1 >= 0) && (ih1 < IH) && (iw1 >= 0) && (iw1 < IW);
		bool lx2 = (ih2 >= 0) && (ih2 < IH) && (iw2 >= 0) && (iw2 < IW);
		bool lx3 = (ih3 >= 0) && (ih3 < IH) && (iw3 >= 0) && (iw3 < IW);

		float4 xv0 = (lx0 ? *(float4*)(X + xoffset0) : F32_4_0);//[j0 <n0, dh0, dw0>, c0 - c3]
		float4 xv1 = (lx1 ? *(float4*)(X + xoffset1) : F32_4_0);//[j1 <n1, dh1, dw1>, c0 - c3]
		float4 xv2 = (lx2 ? *(float4*)(X + xoffset2) : F32_4_0);//[j2 <n2, dh2, dw2>, c0 - c3]
		float4 xv3 = (lx3 ? *(float4*)(X + xoffset3) : F32_4_0);//[j3 <n3, dh3, dw3>, c0 - c3]
		
		//======[write: 4*4 elements to D]=========================================================
		int doffset0 = c0 * GM + j0;
		int doffset1 = c1 * GM + j0;
		int doffset2 = c2 * GM + j0;
		int doffset3 = c3 * GM + j0;

		//transpose: [j0 - j3, c0 - c3] -> [c0 - c3, j0 - j3]
		*(float4*)(D + doffset0) = float4{ xv0.x, xv1.x, xv2.x, xv3.x };//[c0, j0 - j3]
		*(float4*)(D + doffset1) = float4{ xv0.y, xv1.y, xv2.y, xv3.y };//[c1, j0 - j3]
		*(float4*)(D + doffset2) = float4{ xv0.z, xv1.z, xv2.z, xv3.z };//[c2, j0 - j3]
		*(float4*)(D + doffset3) = float4{ xv0.w, xv1.w, xv2.w, xv3.w };//[c3, j0 - j3]
	}
}

#endif


#ifndef CONV_3D_INPUT_PAD_REMORE_KERNEL_4_4S
#define CONV_3D_INPUT_PAD_REMORE_KERNEL_4_4S

//ph = pw = 1:
//[N, IH, IW, IC] = [ 64, 32, 32, 128]: 305.535 GB/s
//[N, IH, IW, IC] = [256, 16, 16, 128]: 288.556 GB/s
//[N, IH, IW, IC] = [256,  8,  8, 256]: 231.675 GB/s
//[N, IH, IW, IC] = [512,  4,  4, 512]: 161.436 GB/s

template<int LB>
__global__ void input_pad_remode_kernel_4_4S(
	const float* __restrict__ X, int IH, int IW,
	      float* __restrict__ D, int DH_DW, int DW,
	int GM, int IC, int ph, int pw)
{
	const int bx = (blockIdx.x << LB);//IC / 4
	const int by = (blockIdx.y << LB);//(N * DN * DW) / 4
	const int tx = threadIdx.x, ty = threadIdx.y;

	const int stepX = (gridDim.x << LB), stepX4 = stepX << 2;
	const int stepY = (gridDim.y << LB), stepY4 = stepY << 2;

	__shared__ float4 Ds[4][(1 << LB) + 1][(1 << LB) + 1];
	
	for (int bx4 = bx << 2; bx4 < IC; bx4 += stepX4)//y4 -> [ic0 -> ic4]
	for (int by4 = by << 2; by4 < GM; by4 += stepY4)//x4 -> [j0 - j4] : (n, ih, iw)
	{
		//======[load: 4*4 elements from X]========================================================
		const int rx4 = bx4 + (tx << 2);//ic
		const int ry4 = by4 + (ty << 2); {//j: (n, dh, dw)
			const int c0 = rx4;
			const int j0 = ry4, j1 = j0 + 1, j2 = j0 + 2, j3 = j0 + 3;

			int n0, dh0, dw0; { n0 = j0 / DH_DW; int jr = j0 - n0 * DH_DW; dh0 = jr / DW; dw0 = jr - dh0 * DW; }
			int n1, dh1, dw1; { n1 = j1 / DH_DW; int jr = j1 - n1 * DH_DW; dh1 = jr / DW; dw1 = jr - dh1 * DW; }
			int n2, dh2, dw2; { n2 = j2 / DH_DW; int jr = j2 - n2 * DH_DW; dh2 = jr / DW; dw2 = jr - dh2 * DW; }
			int n3, dh3, dw3; { n3 = j3 / DH_DW; int jr = j3 - n3 * DH_DW; dh3 = jr / DW; dw3 = jr - dh3 * DW; }

			int ih0 = dh0 - ph, iw0 = dw0 - pw;
			int ih1 = dh1 - ph, iw1 = dw1 - pw;
			int ih2 = dh2 - ph, iw2 = dw2 - pw;
			int ih3 = dh3 - ph, iw3 = dw3 - pw;

			int xoffset0 = ((n0*IH + ih0)*IW + iw0)*IC + c0;//[c0 - c3], IC % 4 == 0, c0 % 4 == 0
			int xoffset1 = ((n1*IH + ih1)*IW + iw1)*IC + c0;//[c0 - c3]
			int xoffset2 = ((n2*IH + ih2)*IW + iw2)*IC + c0;//[c0 - c3]
			int xoffset3 = ((n3*IH + ih3)*IW + iw3)*IC + c0;//[c0 - c3]

			bool lx0 = (ih0 >= 0) && (ih0 < IH) && (iw0 >= 0) && (iw0 < IW);
			bool lx1 = (ih1 >= 0) && (ih1 < IH) && (iw1 >= 0) && (iw1 < IW);
			bool lx2 = (ih2 >= 0) && (ih2 < IH) && (iw2 >= 0) && (iw2 < IW);
			bool lx3 = (ih3 >= 0) && (ih3 < IH) && (iw3 >= 0) && (iw3 < IW);

			if (rx4 < IC && ry4 < GM) {
				float4 xv0 = (lx0 ? *(float4*)(X + xoffset0) : F32_4_0);//[j0 <n0, dh0, dw0>, c0 - c3]
				float4 xv1 = (lx1 ? *(float4*)(X + xoffset1) : F32_4_0);//[j1 <n1, dh1, dw1>, c0 - c3]
				float4 xv2 = (lx2 ? *(float4*)(X + xoffset2) : F32_4_0);//[j2 <n2, dh2, dw2>, c0 - c3]
				float4 xv3 = (lx3 ? *(float4*)(X + xoffset3) : F32_4_0);//[j3 <n3, dh3, dw3>, c0 - c3]

				Ds[0][ty][tx] = xv0;
				Ds[1][ty][tx] = xv1;
				Ds[2][ty][tx] = xv2;
				Ds[3][ty][tx] = xv3;
			}
		}
		__syncthreads();
		
		//======[write: 4*4 elements to D]=========================================================
		const int wx4 = bx4 + (ty << 2);//ic
		const int wy4 = by4 + (tx << 2); {//j: (n, dh, dw)
			const int c0 = wx4, c1 = c0 + 1, c2 = c0 + 2, c3 = c0 + 3;
			const int j0 = wy4;

			int doffset0 = c0 * GM + j0;//(c0: y, j0: x)
			int doffset1 = c1 * GM + j0;
			int doffset2 = c2 * GM + j0;
			int doffset3 = c3 * GM + j0;

			float4 xv0 = Ds[0][tx][ty];
			float4 xv1 = Ds[1][tx][ty];
			float4 xv2 = Ds[2][tx][ty];
			float4 xv3 = Ds[3][tx][ty];

			if (wx4 < IC && wy4 < GM) {
				//transpose: [j0 - j3, c0 - c3] -> [c0 - c3, j0 - j3]
				*(float4*)(D + doffset0) = float4{ xv0.x, xv1.x, xv2.x, xv3.x };//[c0, j0 - j3]
				*(float4*)(D + doffset1) = float4{ xv0.y, xv1.y, xv2.y, xv3.y };//[c1, j0 - j3]
				*(float4*)(D + doffset2) = float4{ xv0.z, xv1.z, xv2.z, xv3.z };//[c2, j0 - j3]
				*(float4*)(D + doffset3) = float4{ xv0.w, xv1.w, xv2.w, xv3.w };//[c3, j0 - j3]
			}
		}
		__syncthreads();
	}
}

#endif


//======[integration]==============================================
#ifndef CONV_3D_INPUT_PAD_REMORE_FUNCTION
#define CONV_3D_INPUT_PAD_REMORE_FUNCTION

//<1> IC % 4 == 0
//<2> N  % 4 == 0 -> GM % 4 == 0
void __conv3D_input_pad_remode(cudaStream_t stream,
	const float* X, int IH, int IW,
	      float* D, int DH, int DW,
	int N, int IC, int ph, int pw)
{
	//mat transpose approach (4*4)--------------------------------------------
	int GM = N * DH * DW;//{ LTY, LTX } >= 2

	if( IC > 63 && GM > 255) { input_pad_remode_k44S(stream, 4, 2, X, IH, IW, D, DH, DW, N, IC, ph, pw, GM); return; }//(5, 4), LB = 5
	if (IC > 31 && GM > 127) { input_pad_remode_k44S(stream, 3, 2, X, IH, IW, D, DH, DW, N, IC, ph, pw, GM); return; }//(5, 4), LB = 5

	if (IC > 31 && GM > 15) { input_pad_remode_k44(stream, 3, 2, 2, 2, X, IH, IW, D, DH, DW, N, IC, ph, pw, GM); return; }//(5, 4), LB = 5
	if (IC > 15 && GM > 31) { input_pad_remode_k44(stream, 2, 3, 2, 2, X, IH, IW, D, DH, DW, N, IC, ph, pw, GM); return; }//(4, 5), LB = 5
	if (IC > 63 && GM >  7) { input_pad_remode_k44(stream, 4, 1, 2, 2, X, IH, IW, D, DH, DW, N, IC, ph, pw, GM); return; }//(6, 3), LB = 5
	if (IC >  7 && GM > 63) { input_pad_remode_k44(stream, 1, 4, 2, 2, X, IH, IW, D, DH, DW, N, IC, ph, pw, GM); return; }//(3, 6), LB = 5
	if (IC > 127          ) { input_pad_remode_k44(stream, 5, 0, 2, 2, X, IH, IW, D, DH, DW, N, IC, ph, pw, GM); return; }//(7, 2), LB = 5
	if (          GM > 127) { input_pad_remode_k44(stream, 0, 5, 2, 2, X, IH, IW, D, DH, DW, N, IC, ph, pw, GM); return; }//(2, 7), LB = 5
	if (IC > 15 && GM > 15) { input_pad_remode_k44(stream, 2, 2, 2, 2, X, IH, IW, D, DH, DW, N, IC, ph, pw, GM); return; }//(4, 4), LB = 4
	if (IC > 31 && GM >  7) { input_pad_remode_k44(stream, 3, 1, 2, 2, X, IH, IW, D, DH, DW, N, IC, ph, pw, GM); return; }//(5, 3), LB = 4
	if (IC >  7 && GM > 31) { input_pad_remode_k44(stream, 1, 3, 2, 2, X, IH, IW, D, DH, DW, N, IC, ph, pw, GM); return; }//(3, 5), LB = 4
	if (IC > 63           ) { input_pad_remode_k44(stream, 4, 0, 2, 2, X, IH, IW, D, DH, DW, N, IC, ph, pw, GM); return; }//(7, 2), LB = 4
	if (           GM > 63) { input_pad_remode_k44(stream, 0, 4, 2, 2, X, IH, IW, D, DH, DW, N, IC, ph, pw, GM); return; }//(2, 7), LB = 4

	//normal approach--------------------------------------------------------
	int lengthv = N * DH * DW * IC;
	if (lengthv < 256) { input_pad_remode_k4_small(stream, X, IH, IW, D, DH, DW, N, IC, ph, pw, lengthv); return; }
	input_pad_remode_k4(stream, 5, 2, X, IH, IW, D, DH, DW, N, IC, ph, pw, lengthv);
}


inline void __deconv3D_input_pad_remode(cudaStream_t stream,
	const float* deltaY, int OH, int OW,
	float*      D, int DH, int DW,
	int N, int OC, int ph, int pw) {
	__conv3D_input_pad_remode(stream,
		deltaY, OH, OW,
		D, DH, DW,
		N, OC, ph, pw);
}

#endif

#endif