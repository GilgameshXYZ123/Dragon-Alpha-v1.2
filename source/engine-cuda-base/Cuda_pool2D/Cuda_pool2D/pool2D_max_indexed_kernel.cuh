#pragma once

#ifndef POOL2D_MAX_INDEXED_KERNEL_H
#define POOL2D_MAX_INDEXED_KERNEL_H

//(1) FH * FW >= 2;
//(2) GN = IC; GN % 4 == 0, GN >= 4
//(3) GM = N * OH * OW;
//(4) GK = FH * FW >= 2
//(5) the memory padding alignment only effects N, IC, OC
#ifndef POOL2D_MAX_INDEXED_KERNEL_CALL
#define POOL2D_MAX_INDEXED_KERNEL_CALL

//======[extra]=============================================================
#define kmaxIdx4_div2(stream, LBY, LBX, X, IH, IW, Y, Index, OH, OW, GN, GM, ic_index, j_index)\
	max_indexed_div2_kernel_4\
		<<< dim3(GM>>LBX , GN>>LBY>>2), dim3(1<<LBX, 1<<LBY), 0, stream>>>\
			(X, IH, IW, Y, Index, (OH*OW), OW, IC, ic_index, j_index)

#define kmaxIdx4_W3(stream, LBY, LBX, X, IH, IW, Y, Index, OH, OW, sh, sw, ph, pw, GN, GM, ic_index, j_index)\
	max_indexed_W3_kernel_4\
		<<< dim3(GM>>LBX , GN>>LBY>>2), dim3(1<<LBX, 1<<LBY), 0, stream>>>\
			(X, IH, IW, Y, Index, (OH*OW), OW, IC, sh, sw, ph, pw, ic_index, j_index)

//======[common]============================================================
#define kmaxIdx4(stream, LBY, LBX, X, IH, IW, FH, FW, Y, Index, OH, OW, sh, sw, ph, pw, GN, GM, ic_index, j_index)\
	max_indexed_kernel_4\
		<<< dim3(GM>>LBX , GN>>LBY>>2), dim3(1<<LBX, 1<<LBY), 0, stream>>>\
			(X, IH, IW, FH, FW, Y, Index, (OH*OW), OW, IC, sh, sw, ph, pw, ic_index, j_index)

#define kmaxIdx2(stream, LBY, LBX, X, IH, IW, FH, FW, Y, Index, OH, OW, sh, sw, ph, pw, GN, GM, ic_index, j_index)\
	max_indexed_kernel_2\
		<<< dim3(GM>>LBX , GN>>LBY>>1), dim3(1<<LBX, 1<<LBY), 0, stream>>>\
			(X, IH, IW, FH, FW, Y, Index, OH, OW, IC, sh, sw, ph, pw, ic_index, j_index)

#define kmaxIdx1(stream, LBY, LBX, X, IH, IW, FH, FW, Y, Index, OH, OW, sh, sw, ph, pw, GN, GM, ic_index, j_index)\
	max_indexed_kernel_1\
		<<< dim3(GM>>LBX , GN>>LBY), dim3(1<<LBX, 1<<LBY), 0, stream>>>\
			(X, IH, IW, FH, FW, Y, Index, OH, OW, IC, sh, sw, ph, pw, ic_index, j_index)

#endif


//======[extra]=============================================================
//(Y: BLOCK_SIZE * 4): div = 2, sh = sw = FH = FW = 2, ph = pw = 0
#ifndef POOL2D_MAX_DIV2_INDEXED_KERNEL_4
#define POOL2D_MAX_DIV2_INDEXED_KERNEL_4

//<4, 1>: Size = 0.125000, Time = 1.610000 msec, Performance = 83.365044 GFlop/s
__global__ void max_indexed_div2_kernel_4(
	const float* __restrict__ X, int IH, int IW,
	      float* __restrict__ Y,
	int* __restrict__ Index, int OH_OW, int OW,
	int IC,//FH = FW = 2
	//ph = pw = 0, sh = sw = 2
	int ic_index, int j_index)
{
	const int bx = blockIdx.x, by = blockIdx.y;
	const int tx = threadIdx.x, ty = threadIdx.y;

	float4 v = F32_MIN4; int4 idx = I32_4_0;//compute 1*4 elements

	//prepare for GN = IC
	const int ic = (((by * blockDim.y) + ty) << 2) + ic_index;

	//prepare for GM = N * OH * OW
	const int j = ((bx * blockDim.x) + tx) + j_index;
	get_n_oh_ow(j, n, oh, ow);
	const int toh = oh << 1, tow = ow << 1;
	const int shiftX = ((n*IH + toh)*IW + tow) * IC + ic;//X[n, toh, tow, ic]
	X += shiftX;

	const int xoffset0 = 0;            //[fh, fw] = [0, 0]
	const int xoffset1 = IC;           //[fh, fw] = [0, 1]
	const int xoffset2 = IW * IC;      //[fh, fw] = [1, 0]
	const int xoffset3 = xoffset2 + IC;//[fh, fw] = [1, 1]

	float4 x0 = *(float4*)(X + xoffset0);
	float4 x1 = *(float4*)(X + xoffset1);
	float4 x2 = *(float4*)(X + xoffset2);
	float4 x3 = *(float4*)(X + xoffset3);

	//update the Index first, then the value
	idx.x = IF_int((x0.x > v.x), (xoffset0    ), idx.x);//ic0
	idx.y = IF_int((x0.y > v.y), (xoffset0 + 1), idx.y);//ic1
	idx.z = IF_int((x0.z > v.z), (xoffset0 + 2), idx.z);//ic2
	idx.w = IF_int((x0.w > v.w), (xoffset0 + 3), idx.w);//ic3
	simdMAX4(v, v, x0);

	idx.x = IF_int((x1.x > v.x), (xoffset1    ), idx.x);//ic0
	idx.y = IF_int((x1.y > v.y), (xoffset1 + 1), idx.y);//ic1
	idx.z = IF_int((x1.z > v.z), (xoffset1 + 2), idx.z);//ic2
	idx.w = IF_int((x1.w > v.w), (xoffset1 + 3), idx.w);//ic3
	simdMAX4(v, v, x1);
	
	idx.x = IF_int((x2.x > v.x), (xoffset2    ), idx.x);//ic0
	idx.y = IF_int((x2.y > v.y), (xoffset2 + 1), idx.y);//ic1
	idx.z = IF_int((x2.z > v.z), (xoffset2 + 2), idx.z);//ic2
	idx.w = IF_int((x2.w > v.w), (xoffset2 + 3), idx.w);//ic3
	simdMAX4(v, v, x2);
	
	idx.x = IF_int((x3.x > v.x), (xoffset3    ), idx.x);//ic0
	idx.y = IF_int((x3.y > v.y), (xoffset3 + 1), idx.y);//ic1
	idx.z = IF_int((x3.z > v.z), (xoffset3 + 2), idx.z);//ic2
	idx.w = IF_int((x3.w > v.w), (xoffset3 + 3), idx.w);//ic3
	simdMAX4(v, v, x3);
	
	idx.x += shiftX;
	idx.y += shiftX;
	idx.z += shiftX;
	idx.w += shiftX;

	const int yoffset = j * IC + ic;
	*(float4*)(Y + yoffset) = v;
	*(int4*)(Index + yoffset) = idx;
}

#endif


//(Y: BLOCK_SIZE * 4)
#ifndef POOL2D_MAX_INDEXED_W3_KERNEL_4
#define POOL2D_MAX_INDEXED_W3_KERNEL_4

//<2, 2>: Size = 0.062500, Time = 1.538000 msec, Performance = 43.633850 GFlop/s [ 16, 4]
//<2, 3>: Size = 0.062500, Time = 1.402000 msec, Performance = 47.866524 GFlop/s [ 16, 8]
//<3, 2>: Size = 0.125000, Time = 2.312000 msec, Performance = 58.052647 GFlop/s [ 32, 4]
//<4, 1>: Size = 0.062500, Time = 1.110000 msec, Performance = 60.458431 GFlop/s [ 64, 2]
//<5, 0>: Size = 0.125000, Time = 2.024000 msec, Performance = 66.313103 GFlop/s [128, 1]
__global__ void max_indexed_W3_kernel_4(
	const float* __restrict__ X, int IH, int IW,
	      float* __restrict__ Y,
	int* __restrict__ Index, int OH_OW, int OW,
	int IC,//FH = FW = 3
	int sh, int sw, int ph, int pw,
	int ic_index, int j_index)
{
	const int bx = blockIdx.x, by = blockIdx.y;
	const int tx = threadIdx.x, ty = threadIdx.y;

	float4 v = F32_MIN4; int4 idx = I32_4_0;//compute 1*4 elements

	//prepare for GN = IC
	const int ic = (((by*blockDim.y) + ty) << 2) + ic_index;

	//prepare for GM = N * OH * OW
	const int j = ((bx * blockDim.x) + tx) + j_index;
	get_n_oh_ow(j, n, oh, ow);
	const int toh = oh * sh - ph;
	const int tow = ow * sw - pw;
	const int shiftX = ((n*IH + toh)*IW + tow) * IC + ic;//X[n, toh, tow, ic]
	X += shiftX;

#pragma unroll
	for (int fh = 0; fh < 3; fh++)
	{
		bool lflag = (toh >= -fh) && (toh < IH - fh);
#pragma unroll
		for (int fw = 0; fw < 3; fw++)
		{
			const int xoffset = (fh*IW + fw)*IC;
			bool lx = lflag && (tow >= -fw) && (tow < IW - fw);
			float4 x = (lx ? *(float4*)(X + xoffset) : F32_MIN4);

			//compute index: mapping X -> Y
			idx.x = IF_int((x.x > v.x), (xoffset    ), idx.x);//ic0
			idx.y = IF_int((x.y > v.y), (xoffset + 1), idx.y);//ic1
			idx.z = IF_int((x.z > v.z), (xoffset + 2), idx.z);//ic2
			idx.w = IF_int((x.w > v.w), (xoffset + 3), idx.w);//ic3

			simdMAX4(v, v, x);
		}
	}

	idx.x += shiftX;
	idx.y += shiftX;
	idx.z += shiftX;
	idx.w += shiftX;

	const int yoffset = j * IC + ic;
	*(float4*)(Y + yoffset) = v;
	*(int4*)(Index + yoffset) = idx;
}

#endif


//======[common]============================================================
//(Y: BLOCK_SIZE * 4)
#ifndef POOL2D_MAX_INDEXED_KERNEL_4
#define POOL2D_MAX_INDEXED_KERNEL_4

//<2, 2>: Size = 0.062500, Time = 1.538000 msec, Performance = 43.633850 GFlop/s [ 16, 4]
//<2, 3>: Size = 0.062500, Time = 1.402000 msec, Performance = 47.866524 GFlop/s [ 16, 8]
//<3, 2>: Size = 0.125000, Time = 2.312000 msec, Performance = 58.052647 GFlop/s [ 32, 4]
//<4, 1>: Size = 0.062500, Time = 1.110000 msec, Performance = 60.458431 GFlop/s [ 64, 2]
//<5, 0>: Size = 0.125000, Time = 2.024000 msec, Performance = 66.313103 GFlop/s [128, 1]
__global__ void max_indexed_kernel_4(
	const float* __restrict__ X, int IH, int IW,
	int FH, int FW,
	float* __restrict__ Y, 
	int* __restrict__ Index, int OH_OW, int OW,
	int IC,
	int sh, int sw, int ph, int pw,
	int ic_index, int j_index)
{
	const int bx = blockIdx.x, by = blockIdx.y;
	const int tx = threadIdx.x, ty = threadIdx.y;

	float4 v = F32_MIN4; int4 idx = I32_4_0;//compute 1*4 elements

	//prepare for GN = IC
	const int ic = (((by*blockDim.y) + ty) << 2) + ic_index;

	//prepare for GM = N * OH * OW
	const int j = ((bx * blockDim.x) + tx) + j_index;
	get_n_oh_ow(j, n, oh, ow);
	const int toh = oh * sh - ph;
	const int tow = ow * sw - pw;
	const int shiftX = ((n*IH + toh)*IW + tow) * IC + ic;//X[n, toh, tow, ic]
	X += shiftX;

	for (int fh = 0; fh < FH; fh++)
	{
		bool lflag = (toh >= -fh) && (toh < IH - fh);
		for (int fw = 0; fw < FW; fw++)
		{
			const int xoffset = (fh*IW + fw)*IC;
			bool lx = lflag && (tow >= -fw) && (tow < IW - fw);
			float4 x = (lx ? *(float4*)(X + xoffset) : F32_MIN4);

			//compute index: mapping X -> Y
			idx.x = IF_int((x.x > v.x), (xoffset    ), idx.x);//ic0
			idx.y = IF_int((x.y > v.y), (xoffset + 1), idx.y);//ic1
			idx.z = IF_int((x.z > v.z), (xoffset + 2), idx.z);//ic2
			idx.w = IF_int((x.w > v.w), (xoffset + 3), idx.w);//ic3

			simdMAX4(v, v, x);
		}
	}

	idx.x += shiftX;
	idx.y += shiftX;
	idx.z += shiftX;
	idx.w += shiftX;

	const int yoffset = j * IC + ic;
	*(float4*)(Y + yoffset) = v;
	*(int4*)(Index + yoffset) = idx;
}

#endif


//(Y: BLOCK_SIZE * 2)
#ifndef POOL2D_MAX_INDEXED_KERNEL_2
#define POOL2D_MAX_INDEXED_KERNEL_2

//<2, 2>: Size = 0.062500, Time = 2.796000 msec, Performance = 24.001738 GFlop/s [ 4, 8]
//<2, 3>: Size = 0.062500, Time = 1.862000 msec, Performance = 36.041279 GFlop/s [ 8, 8]
//<3, 2>: Size = 0.062500, Time = 1.740000 msec, Performance = 38.568310 GFlop/s [16, 4]
//<4, 1>: Size = 0.062500, Time = 1.626000 msec, Performance = 41.272358 GFlop/s [32, 2]
//<5, 0>: Size = 0.062500, Time = 1.516000 msec, Performance = 44.267056 GFlop/s [64, 1]
__global__ void max_indexed_kernel_2(
	const float* __restrict__ X, int IH, int IW,
	int FH, int FW,
	float* __restrict__ Y, 
	int* __restrict__ Index, int OH, int OW,
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
	const int shiftX = n * IH * IW * IC;
	X += shiftX;

	float2 v = F32_MIN2;
	int2 idx = make_int2(0, 0);
	for (int fh = 0; fh < FH; fh++)
	{
		int ih = toh + fh;
		bool lflag = (ih >= 0) && (ih < IH);
		for (int fw = 0; fw < FW; fw++)
		{
			int iw = tow + fw;
			if (lflag && iw >= 0 && iw < IW) 
			{
				int xoffset = (ih*IW + iw)*IC + ic;
				float2 x = *(float2*)(X + xoffset);

				//compute index: which X -> Y
				//if: x  > v: idx = (idx - xoffset)*0 + xoffset = xoffset
				//if: x <= v: idx = (idx - xoffset)*1 + xoffset = idx
				idx.x = (idx.x - (xoffset    )) * (x.x < v.x) + (xoffset    );
				idx.y = (idx.y - (xoffset + 1)) * (x.y < v.y) + (xoffset + 1);

				simdMAX2(v, v, x);
			}
		}
	}

	idx.x += shiftX;
	idx.y += shiftX;

	int yoffset = j * IC + ic;
	*(float2*)(Y + yoffset) = v;
	*(int2*)(Index + yoffset) = idx;
}

#endif


//(Y: BLOCK_SIZE * 1)
#ifndef POOL2D_MAX_INDEXED_KERNEL_1
#define POOL2D_MAX_INDEXED_KERNEL_1

//<2, 2>: Size = 0.062500, Time = 5.292000 msec, Performance = 12.681190 GFlop/s [ 4, 4]
//<2, 3>: Size = 0.062500, Time = 3.634000 msec, Performance = 18.466940 GFlop/s [ 4, 8]
//<3, 2>: Size = 0.062500, Time = 2.842000 msec, Performance = 23.613251 GFlop/s [ 8, 4]
//<4, 1>: Size = 0.062500, Time = 2.562000 msec, Performance = 26.193933 GFlop/s [16, 2]
//<5, 0>: Size = 0.062500, Time = 2.664000 msec, Performance = 25.191011 GFlop/s [32, 1]
__global__ void max_indexed_kernel_1(
	const float* __restrict__ X, int IH, int IW,
	int FH, int FW,
	float* __restrict__ Y, 
	int* __restrict__ Index, int OH, int OW,
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
	const int shiftX = n * IH * IW * IC;
	X += shiftX;

	float v = FLOAT_MIN;
	int idx = 0;
	for (int fh = 0; fh < FH; fh++)
	{
		int ih = toh + fh;
		bool lflag = (ih >= 0) && (ih < IH);
		for (int fw = 0; fw < FW; fw++)
		{
			int iw = tow + fw;
			if (lflag && iw >= 0 && iw < IW) 
			{
				int xoffset = (ih*IW + iw)*IC + ic;
				float x = X[xoffset];

				//compute index: which X -> Y
				//if: x  > v: idx = (idx - xoffset)*0 + xoffset = xoffset
				//if: x <= v: idx = (idx - xoffset)*1 + xoffset = idx
				idx = (idx - (xoffset)) * (x < v) + (xoffset);
				
				v = fmaxf(v, x);
			}
		}
	}

	idx += shiftX;

	int yoffset = j * IC + ic;
	Y[yoffset] = v;
	Index[yoffset] = idx;
}

#endif

#endif