


//(Y: BLOCK_SIZE * 4)
#ifndef MV3_KERNEL1
#define MV3_KERNEL1

#define mv3k1(stream, LBY, LBX, X, IH, IW, FH, FW, Y, OH, OW, sh, sw, ph, pw, GN, GM, ic_index, j_index)\
	MV3_kernel1\
		<<< dim3(GM>>LBX , GN>>LBY>>2), dim3(1<<LBX, 1<<LBY), 0, stream>>>\
			(X, IH, IW, Y, (OH*OW), OW, IC, sh, sw, ph, pw, ic_index, j_index)

//for: FH * FW = 3*3
//<4, 1>: Size = 0.281250, Time = 1.626000 msec, Performance = 185.725616 GFlop/s
__global__ void MV3_kernel1(
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



//div = 2
#ifndef MV3_KERNEL2
#define MV3_KERNEL2

#define mv3k2(stream, LBY, LBX, X, IH, IW, FH, FW, Y, OH, OW, sh, sw, ph, pw, GN, GM, ic_index, j_index)\
	MV3_kernel2\
		<<< dim3(GM>>LBX , GN>>LBY>>2), dim3(1<<LBX, 1<<LBY), 0, stream>>>\
			(X, IH, IW, Y, (OH*OW), OW, IC, ic_index, j_index)

//Size = 0.125000, Time = 1.610000 msec, Performance = 83.365044 GFlop/s

__global__ void MV3_kernel2(
	const float* __restrict__ X, int IH, int IW,
	float* __restrict__ Y, int OH_OW, int OW,
	int IC,//FH = FW = 2, ph = pw = 0, sh = sw = 2
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

	const int xoffset0 =  0;           //[fh, fw] = [0, 0]
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