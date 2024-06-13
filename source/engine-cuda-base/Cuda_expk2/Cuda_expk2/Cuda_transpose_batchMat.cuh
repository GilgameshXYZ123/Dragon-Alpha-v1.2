#pragma once

#ifndef BATCH_MAT_TRANSPOSE_H
#define BATCH_MAT_TRANSPOSE_H

//like a batch_transpose2D: for(dim0, [dim1, dim2]) 
//batch_matrixTranspose: [batch, N, M]
//only used for transpose(-1, -2), and: dim(-2) % 4 == 0
//Batch = mul(dim[0] -> dim[-3])
#ifndef BATCH_MAT_TRANSPOSE_KERNEL_CALL
#define BATCH_MAT_TRANSPOSE_KERNEL_CALL

//LTY>>1, LTX>>1
#define batch_matTrans_k22(stream, LBY, LBX, LTY, LTX, A, AT, Batch, N, M) \
	batch_mat_transpose_kernel_2_2\
		<<< dim3((M>>LBX>>LTX), (N>>LBY>>LTY), ((Batch+7)>>3)),\
		    dim3(1<<LBX, 1<<LBY, 8), 0, stream >>>\
				(A, AT, Batch, N, M)

//LTY>>2, LTX>>2
#define batch_matTrans_k44(stream, LBY, LBX, LTY, LTX, A, AT, Batch, N, M) \
	batch_mat_transpose_kernel_4_4\
		<<< dim3((M>>LBX>>LTX), (N>>LBY>>LTY), ((Batch+7)>>3)),\
			 dim3(1<<LBX, 1<<LBY, 8), 0, stream >>>\
				(A, AT, Batch, N, M)

#endif


//N % 4 == 0, M % 4 == 0
#ifndef BATCH_MAT_TRANSPOSE_KERNEL_4_4
#define BATCH_MAT_TRANSPOSE_KERNEL_4_4

//Size = 32, Time = 0.448 mesc, Speed = 69.7545GB/s
__global__ void batch_mat_transpose_kernel_4_4(
	const float* __restrict__ A,	
	      float* __restrict__ AT,
	int Batch, int N, int M)
{
	int B = (blockIdx.z*blockDim.z) + threadIdx.z;
	int Y = (blockIdx.y*blockDim.y) + threadIdx.y;
	int X = (blockIdx.x*blockDim.x) + threadIdx.x;
	int Boffset = B * N * M; A += Boffset; AT += Boffset;

	int stepB = (blockDim.z*gridDim.z), strideA = N * M * stepB;
	int stepY4 = (blockDim.y*gridDim.y) << 2;
	int stepX4 = (blockDim.x*gridDim.x) << 2;

	float4 a0, a1, a2, a3;
	for (; B < Batch; B += stepB)//mul(dim[0] -> dim[-3])
	{
		for (int Y4 = (Y << 2); Y4 < N; Y4 += stepY4)
		for (int X4 = (X << 2); X4 < M; X4 += stepX4)
		{
			a0 = *(float4*)(&get2d(A, Y4	, X4, M));
			a1 = *(float4*)(&get2d(A, Y4 + 1, X4, M));
			a2 = *(float4*)(&get2d(A, Y4 + 2, X4, M));
			a3 = *(float4*)(&get2d(A, Y4 + 3, X4, M));

			//inner transpose: 4 * 4 = 16
			*(float4*)(&get2d(AT, X4    , Y4, N)) = make_float4(a0.x, a1.x, a2.x, a3.x);
			*(float4*)(&get2d(AT, X4 + 1, Y4, N)) = make_float4(a0.y, a1.y, a2.y, a3.y);
			*(float4*)(&get2d(AT, X4 + 2, Y4, N)) = make_float4(a0.z, a1.z, a2.z, a3.z);
			*(float4*)(&get2d(AT, X4 + 3, Y4, N)) = make_float4(a0.w, a1.w, a2.w, a3.w);
		}
		A += strideA; AT += strideA;
	}
}

#endif


//N % 2 == 0, M % 2 == 0
#ifndef BATCH_MAT_TRANSPOSE_KERNEL_2_2
#define BATCH_MAT_TRANSPOSE_KERNEL_2_2

//Size = 32, Time = 0.35 mesc, Speed = 89.2857GB/s
__global__ void batch_mat_transpose_kernel_2_2(
	const float* __restrict__ A,
	float* __restrict__ AT,
	int Batch, int N, int M)
{
	int B = (blockIdx.z*blockDim.z) + threadIdx.z;
	int Y = (blockIdx.y*blockDim.y) + threadIdx.y;
	int X = (blockIdx.x*blockDim.x) + threadIdx.x;
	int Boffset = B * N * M; A += Boffset; AT += Boffset;

	int stepB = (blockDim.z*gridDim.z), strideA = N * M * stepB;
	int stepY2 = (blockDim.y*gridDim.y) << 1;
	int stepX2 = (blockDim.x*gridDim.x) << 1;

	float2 a0, a1;
	for (; B < Batch; B += stepB)//mul(dim[0] -> dim[-3])
	{
		for (int Y2 = (Y << 1); Y2 < N; Y2 += stepY2)
		for (int X2 = (X << 1); X2 < M; X2 += stepX2)
		{
			a0 = *(float2*)(&get2d(A, Y2	, X2, M));
			a1 = *(float2*)(&get2d(A, Y2 + 1, X2, M));

			//inner transpose: 2 * 2 = 4
			*(float2*)(&get2d(AT, X2    , Y2, N)) = make_float2(a0.x, a1.x);
			*(float2*)(&get2d(AT, X2 + 1, Y2, N)) = make_float2(a0.y, a1.y);
		}
		A += strideA; AT += strideA;
	}
}

#endif


//Integrations
#ifndef BATCH_MAT_TRANSPOSE_FUNCTION
#define BATCH_MAT_TRANSPOSE_FUNCTION

bool __batch_mat_transpose(cudaStream_t stream,
	const float* __restrict__ A,
	float* __restrict__ AT,
	int Batch, int N, int M)
{
	if (!(N & 3) && !(M & 3)) {//[LBY, LBX, LTY, LTX], { LTY, LTX } >= 2
		if (N > 31 && M > 15) { batch_matTrans_k44(stream, 3, 2, 2, 2, A, AT, Batch, N, M); return true; }//(5, 4), LB = 5
		if (N > 15 && M > 31) { batch_matTrans_k44(stream, 2, 3, 2, 2, A, AT, Batch, N, M); return true; }//(4, 5), LB = 5
		if (N > 63 && M >  7) { batch_matTrans_k44(stream, 4, 1, 2, 2, A, AT, Batch, N, M); return true; }//(6, 3), LB = 5
		if (N >  7 && M > 63) { batch_matTrans_k44(stream, 1, 4, 2, 2, A, AT, Batch, N, M); return true; }//(3, 6), LB = 5
		if (N > 127         ) { batch_matTrans_k44(stream, 5, 0, 2, 2, A, AT, Batch, N, M); return true; }//(7, 2), LB = 5
		if (M > 127         ) { batch_matTrans_k44(stream, 0, 5, 2, 2, A, AT, Batch, N, M); return true; }//(2, 7), LB = 5
		if (N > 15 && M > 15) { batch_matTrans_k44(stream, 2, 2, 2, 2, A, AT, Batch, N, M); return true; }//(4, 4), LB = 4
		if (N > 15 && M > 15) { batch_matTrans_k44(stream, 2, 2, 2, 2, A, AT, Batch, N, M); return true; }//(4, 4), LB = 4
		if (N > 31 && M >  7) { batch_matTrans_k44(stream, 3, 1, 2, 2, A, AT, Batch, N, M); return true; }//(5, 3), LB = 4
		if (N >  7 && M > 31) { batch_matTrans_k44(stream, 1, 3, 2, 2, A, AT, Batch, N, M); return true; }//(3, 5), LB = 4
		if (N > 63          ) { batch_matTrans_k44(stream, 4, 0, 2, 2, A, AT, Batch, N, M); return true; }//(7, 2), LB = 4
		if (M > 63          ) { batch_matTrans_k44(stream, 0, 4, 2, 2, A, AT, Batch, N, M); return true; }//(2, 7), LB = 4
	}
	if (!(N & 1) && !(M & 1)) {//[LBY, LBX, LTY, LTX], { LTY, LTX } >= 1
		if (N > 15 && M >  7) { batch_matTrans_k22(stream, 3, 2, 1, 1, A, AT, Batch, N, M); return true; }//(4, 3), LB = 5
		if (N >  7 && M > 15) { batch_matTrans_k22(stream, 2, 3, 1, 1, A, AT, Batch, N, M); return true; }//(3, 4), LB = 5 
		if (N > 31          ) { batch_matTrans_k22(stream, 4, 1, 1, 1, A, AT, Batch, N, M); return true; }//(5, 2), LB = 5
		if (M > 31          ) { batch_matTrans_k22(stream, 1, 4, 1, 1, A, AT, Batch, N, M); return true; }//(2, 5), LB = 5 
		if (N >  7 && M >  7) { batch_matTrans_k22(stream, 2, 2, 1, 1, A, AT, Batch, N, M); return true; }//(3, 3), LB = 4
		if (N > 15          ) { batch_matTrans_k22(stream, 3, 1, 1, 1, A, AT, Batch, N, M); return true; }//(4, 2), LB = 4 
		if (M > 15          ) { batch_matTrans_k22(stream, 1, 3, 1, 1, A, AT, Batch, N, M); return true; }//(2, 4), LB = 4
	}
	return false;
}

#endif


//======[when Batch = 1, let: gridDim.z = blockDim.z = 1]=============
#ifndef MAT_TRANSPOSE_KERNEL_CALL
#define MAT_TRANSPOSE_KERNEL_CALL

//LTY>>1, LTX>>1
#define matTrans_k22(stream, LBY, LBX, LTY, LTX, A, AT, N, M) \
	batch_mat_transpose_kernel_2_2\
		<<< dim3((M>>LBX>>LTX), (N>>LBY>>LTY), 1),\
		    dim3(1<<LBX, 1<<LBY, 1), 0, stream >>>\
				(A, AT, 1, N, M)

//LTY>>2, LTX>>2
#define matTrans_k44(stream, LBY, LBX, LTY, LTX, A, AT, N, M) \
	batch_mat_transpose_kernel_4_4\
		<<< dim3((M>>LBX>>LTX), (N>>LBY>>LTY), 1),\
			 dim3(1<<LBX, 1<<LBY, 1), 0, stream >>>\
				(A, AT, 1, N, M)

#endif


#ifndef MAT_TRANSPOSE_FUNCTION
#define MAT_TRANSPOSE_FUNCTION

bool __mat_transpose(cudaStream_t stream,
	const float* __restrict__ A,
	float* __restrict__ AT,
	int N, int M)
{
	if (!(N & 3) && !(M & 3)) {//[LBY, LBX, LTY, LTX], { LTY, LTX } >= 2
		if (N > 31 && M > 15) { matTrans_k44(stream, 3, 2, 2, 2, A, AT, N, M); return true; }//(5, 4), LB = 5
		if (N > 15 && M > 31) { matTrans_k44(stream, 2, 3, 2, 2, A, AT, N, M); return true; }//(4, 5), LB = 5
		if (N > 63 && M >  7) { matTrans_k44(stream, 4, 1, 2, 2, A, AT, N, M); return true; }//(6, 3), LB = 5
		if (N >  7 && M > 63) { matTrans_k44(stream, 1, 4, 2, 2, A, AT, N, M); return true; }//(3, 6), LB = 5
		if (N > 127         ) { matTrans_k44(stream, 5, 0, 2, 2, A, AT, N, M); return true; }//(7, 2), LB = 5
		if (M > 127         ) { matTrans_k44(stream, 0, 5, 2, 2, A, AT, N, M); return true; }//(2, 7), LB = 5
		if (N > 15 && M > 15) { matTrans_k44(stream, 2, 2, 2, 2, A, AT, N, M); return true; }//(4, 4), LB = 4
		if (N > 15 && M > 15) { matTrans_k44(stream, 2, 2, 2, 2, A, AT, N, M); return true; }//(4, 4), LB = 4
		if (N > 31 && M >  7) { matTrans_k44(stream, 3, 1, 2, 2, A, AT, N, M); return true; }//(5, 3), LB = 4
		if (N >  7 && M > 31) { matTrans_k44(stream, 1, 3, 2, 2, A, AT, N, M); return true; }//(3, 5), LB = 4
		if (N > 63          ) { matTrans_k44(stream, 4, 0, 2, 2, A, AT, N, M); return true; }//(7, 2), LB = 4
		if (M > 63          ) { matTrans_k44(stream, 0, 4, 2, 2, A, AT, N, M); return true; }//(2, 7), LB = 4
	}

	if (!(N & 1) && !(M & 1)) {//[LBY, LBX, LTY, LTX], { LTY, LTX } >= 1
		if (N > 15 && M >  7) { matTrans_k22(stream, 3, 2, 1, 1, A, AT, N, M); return true; }//(4, 3), LB = 5
		if (N >  7 && M > 15) { matTrans_k22(stream, 2, 3, 1, 1, A, AT, N, M); return true; }//(3, 4), LB = 5 
		if (N > 31          ) { matTrans_k22(stream, 4, 1, 1, 1, A, AT, N, M); return true; }//(5, 2), LB = 5
		if (M > 31          ) { matTrans_k22(stream, 1, 4, 1, 1, A, AT, N, M); return true; }//(2, 5), LB = 5 
		if (N >  7 && M >  7) { matTrans_k22(stream, 2, 2, 1, 1, A, AT, N, M); return true; }//(3, 3), LB = 4
		if (N > 15          ) { matTrans_k22(stream, 3, 1, 1, 1, A, AT, N, M); return true; }//(4, 2), LB = 4 
		if (M > 15          ) { matTrans_k22(stream, 1, 3, 1, 1, A, AT, N, M); return true; }//(2, 4), LB = 4
	}
	return false;
}

#endif

#endif