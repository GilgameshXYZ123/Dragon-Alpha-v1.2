#pragma once

#ifndef POOL2D_MAX_KERNEL_H
#define POOL2D_MAX_KERNEL_H

//(1) FH * FW >= 2;
//(2) GN = IC; GN % 4 == 0, GN >= 4
//(3) GM = N * OH * OW;
//(4) GK = FH * FW >= 2
//(5) the memory padding alignment only effects N, IC, OC
#ifndef POOL2D_MAX_KERNEL_CALL
#define POOL2D_MAX_KERNEL_CALL

//LB = log2(BLOCK_SIZE)

//======[extra]=============================================================
#define kmax4_div2(stream, LBY, LBX, X, IH, IW, Y, OH, OW, GN, GM, ic_index, j_index)\
	max_div2_kernel_4\
		<<< dim3(GM>>LBX , GN>>LBY>>2), dim3(1<<LBX, 1<<LBY), 0, stream>>>\
			(X, IH, IW, Y, (OH*OW), OW, IC, ic_index, j_index)

#define kmax4_W3(stream, LBY, LBX, X, IH, IW, Y, OH, OW, sh, sw, ph, pw, GN, GM, ic_index, j_index)\
	max_W3_kernel_4\
		<<< dim3(GM>>LBX , GN>>LBY>>2), dim3(1<<LBX, 1<<LBY), 0, stream>>>\
			(X, IH, IW, Y, (OH*OW), OW, IC, sh, sw, ph, pw, ic_index, j_index)

//======[common]============================================================
#define kmax4(stream, LBY, LBX, X, IH, IW, FH, FW, Y, OH, OW, sh, sw, ph, pw, GN, GM, ic_index, j_index)\
	max_kernel_4\
		<<< dim3(GM>>LBX , GN>>LBY>>2), dim3(1<<LBX, 1<<LBY), 0, stream>>>\
			(X, IH, IW, FH, FW, Y, (OH*OW), OW, IC, sh, sw, ph, pw, ic_index, j_index)

#define kmax2(stream, LBY, LBX, X, IH, IW, FH, FW, Y, OH, OW, sh, sw, ph, pw, GN, GM, ic_index, j_index)\
	max_kernel_2\
		<<< dim3(GM>>LBX , GN>>LBY>>1), dim3(1<<LBX, 1<<LBY), 0, stream>>>\
			(X, IH, IW, FH, FW, Y, OH, OW, IC, sh, sw, ph, pw, ic_index, j_index)

#define kmax1(stream, LBY, LBX, X, IH, IW, FH, FW, Y, OH, OW, sh, sw, ph, pw, GN, GM, ic_index, j_index)\
	max_kernel_1\
		<<< dim3(GM>>LBX , GN>>LBY), dim3(1<<LBX, 1<<LBY), 0, stream>>>\
			(X, IH, IW, FH, FW, Y, OH, OW, IC, sh, sw, ph, pw, ic_index, j_index)

#endif


//======[extra]=============================================================
//(Y: BLOCK_SIZE * 4): div = 2, sh = sw = FH = FW = 2, ph = pw = 0
#ifndef POOL2D_MAX_DIV2_KERNEL_4
#define POOL2D_MAX_DIV2_KERNEL_4

//<4, 1>: Size = 0.125000, Time = 1.610000 msec, Performance = 83.365044 GFlop/s
__global__ void max_div2_kernel_4(
	const float* __restrict__ X, int IH, int IW,
	float* __restrict__ Y, int OH_OW, int OW,
	int IC,//FH = FW = 2
	//ph = pw = 0, sh = sw = 2
	int ic_index, int j_index)
{
	const int bx = blockIdx.x, by = blockIdx.y;
	const int tx = threadIdx.x, ty = threadIdx.y;

	float4 v = F32_MIN4;//compute 1*4 elements

	//prepare for GN = IC
	const int ic = (((by * blockDim.y) + ty) << 2) + ic_index;

	//prepare for GM = N * OH * OW
	const int j = (bx * blockDim.x) + tx + j_index;
	get_n_oh_ow(j, n, oh, ow);
	const int toh = oh << 1, tow = ow << 1;
	X += ((n*IH + toh)*IW + tow) * IC + ic;//X[n, toh, tow, ic]

	const int xoffset0 = 0;            //[fh, fw] = [0, 0]
	const int xoffset1 = IC;           //[fh, fw] = [0, 1]
	const int xoffset2 = IW * IC;      //[fh, fw] = [1, 0]
	const int xoffset3 = xoffset2 + IC;//[fh, fw] = [1, 1]

	float4 x0 = *(float4*)(X + xoffset0);
	float4 x1 = *(float4*)(X + xoffset1);
	float4 x2 = *(float4*)(X + xoffset2);
	float4 x3 = *(float4*)(X + xoffset3);

	simdMAX4(v, v, x0);
	simdMAX4(v, v, x1);
	simdMAX4(v, v, x2);
	simdMAX4(v, v, x3);

	const int Y0 = j * IC + ic;
	*(float4*)(Y + Y0) = v;
}

#endif


//(Y: BLOCK_SIZE * 4): FH = FW = 3
#ifndef POOL2D_MAX_W3_KERNEL_4
#define POOL2D_MAX_W3_KERNEL_4

//for [FH * FW = 3 * 3]:
//<4, 1>: Size = 0.281250, Time = 1.626000 msec, Performance = 185.725616 GFlop/s
__global__ void max_W3_kernel_4(
	const float* __restrict__ X, int IH, int IW,
	float* __restrict__ Y, int OH_OW, int OW,
	int IC,//FH = FW = 3
	int sh, int sw, int ph, int pw,
	int ic_index, int j_index)
{
	const int bx = blockIdx.x, by = blockIdx.y;
	const int tx = threadIdx.x, ty = threadIdx.y;

	float4 v = F32_MIN4;//compute 1*4 elements

	//prepare for GN = IC
	const int ic = (((by * blockDim.y) + ty) << 2) + ic_index;

	//prepare for GM = N * OH * OW
	const int j = ((bx * blockDim.x) + tx) + j_index;
	get_n_oh_ow(j, n, oh, ow);
	const int toh = oh * sh - ph;
	const int tow = ow * sw - pw;
	X += ((n*IH + toh)*IW + tow) * IC + ic;//X[n, toh, tow, ic]

#pragma unroll
	for (int fh = 0; fh < 3; fh++) {
#pragma unroll
		for (int fw = 0; fw < 3; fw++)
		{
			const int xoffset = (fh*IW + fw)*IC;
			bool lx = (toh >= -fh) && (toh < IH - fh) && (tow >= -fw) && (tow < IW - fw);
			float4 x = (lx ? *(float4*)(X + xoffset) : F32_MIN4);
			simdMAX4(v, v, x);
		}
	}

	const int Y0 = j * IC + ic;
	*(float4*)(Y + Y0) = v;
}

#endif


//======[common]============================================================
//(Y: BLOCK_SIZE * 4)
#ifndef POOL2D_MAX_KERNEL_4
#define POOL2D_MAX_KERNEL_4

//for: FH*FW = 4*4
//<2, 2>: Size = 0.062500, Time = 0.300000 msec, Performance = 223.696198 GFlop/s [ 16, 4]
//<2, 3>: Size = 0.062500, Time = 0.306000 msec, Performance = 219.310013 GFlop/s [ 16, 8]
//<3, 2>: Size = 0.062500, Time = 0.218000 msec, Performance = 307.838806 GFlop/s [ 32, 4]
//<4, 1>: Size = 0.500000, Time = 1.696000 msec, Performance = 316.551208 GFlop/s [ 64, 2]
//<5, 0>: Size = 0.062500, Time = 0.222000 msec, Performance = 302.292145 GFlop/s [128, 1]
//for: FH * FW = 3*3
//<4, 1>: Size = 0.281250, Time = 1.684000 msec, Performance = 179.328888 GFlop/s
__global__ void max_kernel_4(
	const float* __restrict__ X, int IH, int IW,
	int FH, int FW,
	float* __restrict__ Y, int OH_OW, int OW,
	int IC, 
	int sh, int sw, int ph, int pw,
	int ic_index, int j_index)
{
	const int bx = blockIdx.x, by = blockIdx.y;
	const int tx = threadIdx.x, ty = threadIdx.y;

	float4 v = F32_MIN4;//compute 1*4 elements

	//prepare for GN = IC
	const int ic = (((by * blockDim.y) + ty) << 2) + ic_index;

	//prepare for GM = N * OH * OW
	const int j = (bx * blockDim.x) + tx + j_index;
	get_n_oh_ow(j, n, oh, ow);
	const int toh = oh * sh - ph;
	const int tow = ow * sw - pw;
	X += ((n*IH + toh)*IW + tow) * IC + ic;//X[n, toh, tow, ic]

	for (int fh = 0; fh < FH; fh++) 
	{
		bool lxh = (toh >= -fh) && (toh < IH - fh);
		for (int fw = 0; fw < FW; fw++)
		{
			const int xoffset = (fh*IW + fw)*IC;
			bool lx = lxh && (tow >= -fw) && (tow < IW - fw);
			float4 x = (lx ? *(float4*)(X + xoffset) : F32_MIN4);
			simdMAX4(v, v, x);
		}
	}

	const int Y0 = j * IC + ic;
	*(float4*)(Y + Y0) = v;
}

#endif


//(Y: BLOCK_SIZE * 2)
#ifndef POOL2D_MAX_KERNEL_2
#define POOL2D_MAX_KERNEL_2

//<2, 2>: Size = 0.062500, Time = 2.796000 msec, Performance = 24.001738 GFlop/s [ 4, 8]
//<2, 3>: Size = 0.062500, Time = 1.862000 msec, Performance = 36.041279 GFlop/s [ 8, 8]
//<3, 2>: Size = 0.062500, Time = 1.740000 msec, Performance = 38.568310 GFlop/s [16, 4]
//<4, 1>: Size = 0.062500, Time = 1.626000 msec, Performance = 41.272358 GFlop/s [32, 2]
//<5, 0>: Size = 0.062500, Time = 1.516000 msec, Performance = 44.267056 GFlop/s [64, 1]
__global__ void max_kernel_2(
	const float* __restrict__ X, int IH, int IW,
	int FH, int FW,
	float* __restrict__ Y, int OH, int OW,
	int IC,
	int sh, int sw, int ph, int pw,
	int ic_index, int j_index)
{
	//prepare for GN = IC
	int ic = (((blockIdx.y*blockDim.y) + threadIdx.y) << 1) + ic_index;

	//prepare for GM = N * OH * OW
	int j = (blockIdx.x * blockDim.x) + threadIdx.x + j_index;
	const int OH_OW = OH * OW; get_n_oh_ow(j, n, oh, ow);
	const int toh = oh * sh - ph, tow = ow * sw - pw;
	X += n * IH * IW * IC;

	float2 v = make_float2(FLOAT_MIN, FLOAT_MIN);
	for (int fh = 0; fh < FH; fh++)
	{
		int ih = toh + fh;
		bool lflag = (ih >= 0) && (ih < IH);
		for (int fw = 0; fw < FW; fw++)
		{
			int iw = tow + fw;
			if (lflag && iw >= 0 && iw < IW) {
				float2 x = *(float2*)(&get3d(X, ih, iw, ic, IW, IC));
				simdMAX2(v, v, x);
			}
		}
	}
	*(float2*)(&Y[j*IC + ic]) = v;
}

#endif


//(Y: BLOCK_SIZE * 1)
#ifndef POOL2D_MAX_KERNEL_1
#define POOL2D_MAX_KERNEL_1

//<2, 2>: Size = 0.062500, Time = 5.292000 msec, Performance = 12.681190 GFlop/s [ 4, 4]
//<2, 3>: Size = 0.062500, Time = 3.634000 msec, Performance = 18.466940 GFlop/s [ 4, 8]
//<3, 2>: Size = 0.062500, Time = 2.842000 msec, Performance = 23.613251 GFlop/s [ 8, 4]
//<4, 1>: Size = 0.062500, Time = 2.562000 msec, Performance = 26.193933 GFlop/s [16, 2]
//<5, 0>: Size = 0.062500, Time = 2.664000 msec, Performance = 25.191011 GFlop/s [32, 1]
__global__ void max_kernel_1(
	const float* __restrict__ X, int IH, int IW,
	int FH, int FW,
	float* __restrict__ Y, int OH, int OW,
	int IC,
	int sh, int sw, int ph, int pw,
	int ic_index, int j_index)
{
	//prepare for GN = IC
	int ic = (blockIdx.y*blockDim.y) + threadIdx.y + ic_index;

	//prepare for GM = N * OH * OW
	int j = (blockIdx.x * blockDim.x) + threadIdx.x + j_index;
	const int OH_OW = OH * OW; get_n_oh_ow(j, n, oh, ow);
	const int toh = oh * sh - ph, tow = ow * sw - pw;
	X += n * IH * IW * IC;

	float v = FLOAT_MIN;
	for (int fh = 0; fh < FH; fh++)
	{
		int ih = toh + fh;
		bool lflag = (ih >= 0) && (ih < IH);
		for (int fw = 0; fw < FW; fw++)
		{
			int iw = tow + fw;
			if (lflag && iw >= 0 && iw < IW) {
				float x = get3d(X, ih, iw, ic, IW, IC);
				v = fmaxf(v, x);
			}
		}
	}
	Y[j*IC + ic] = v;
}

#endif

#endif