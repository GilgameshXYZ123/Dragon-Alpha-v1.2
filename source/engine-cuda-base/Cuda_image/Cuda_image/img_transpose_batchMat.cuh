#pragma once

#ifndef IMG_BATCH_MAT_TRANSPOSE_H
#define IMG_BATCH_MAT_TRANSPOSE_H

//like a batch_transpose2D: for(dim0, [dim1, dim2]) 
//batch_matrixTranspose: [batch, N, M]
//only used for transpose(-1, -2), and: dim(-2) % 4 == 0
//Batch = mul(dim[0] -> dim[-3])
#ifndef IMG_BATCH_MAT_TRANSPOSE_KERNEL_CALL
#define IMG_BATCH_MAT_TRANSPOSE_KERNEL_CALL

//LTY>>3, LTX>>3
#define img_batch_matTrans_k88(stream, LBY, LBX, LTY, LTX, A, AT, Batch, N, M) \
	img_batch_mat_transpose_kernel_8_8\
		<<< dim3((M>>LBX>>LTX), (N>>LBY>>LTY), ((Batch+7)>>3)),\
			 dim3(1<<LBX, 1<<LBY, 8), 0, stream >>>\
				(A, AT, Batch, N, M)

//LTY>>2, LTX>>2
#define img_batch_matTrans_k44(stream, LBY, LBX, LTY, LTX, A, AT, Batch, N, M) \
	img_batch_mat_transpose_kernel_4_4\
		<<< dim3((M>>LBX>>LTX), (N>>LBY>>LTY), ((Batch+7)>>3)),\
			 dim3(1<<LBX, 1<<LBY, 8), 0, stream >>>\
				(A, AT, Batch, N, M)

//LTY>>1, LTX>>1
#define img_batch_matTrans_k22(stream, LBY, LBX, LTY, LTX, A, AT, Batch, N, M) \
	img_batch_mat_transpose_kernel_2_2\
		<<< dim3((M>>LBX>>LTX), (N>>LBY>>LTY), ((Batch+7)>>3)),\
		    dim3(1<<LBX, 1<<LBY, 8), 0, stream >>>\
				(A, AT, Batch, N, M)

#endif


//N % 4 == 0, M % 4 == 0
#ifndef IMG_BATCH_MAT_TRANSPOSE_KERNEL_8_8
#define IMG_BATCH_MAT_TRANSPOSE_KERNEL_8_8

//Size = 32, Time = 0.448 mesc, Speed = 69.7545GB/s
__global__ void img_batch_mat_transpose_kernel_8_8(
	const char* __restrict__ A,
	char* __restrict__ AT,
	int Batch, int N, int M)
{
	int B = (blockIdx.z*blockDim.z) + threadIdx.z;
	int Y = (blockIdx.y*blockDim.y) + threadIdx.y;
	int X = (blockIdx.x*blockDim.x) + threadIdx.x;
	int Boffset = B * N * M; A += Boffset; AT += Boffset;

	int stepB = (blockDim.z*gridDim.z), strideA = N * M * stepB;
	int stepY8 = (blockDim.y*gridDim.y) << 3;
	int stepX8 = (blockDim.x*gridDim.x) << 3;

	char8 a0, a1, a2, a3, a4, a5, a6, a7;
	for (; B < Batch; B += stepB)//mul(dim[0] -> dim[-3])
	{
		for (int Y8 = (Y << 3); Y8 < N; Y8 += stepY8)
		for (int X8 = (X << 3); X8 < M; X8 += stepX8)
		{
			a0 = *(char8*)(&get2d(A, Y8    , X8, M));
			a1 = *(char8*)(&get2d(A, Y8 + 1, X8, M));
			a2 = *(char8*)(&get2d(A, Y8 + 2, X8, M));
			a3 = *(char8*)(&get2d(A, Y8 + 3, X8, M));
			a4 = *(char8*)(&get2d(A, Y8 + 4, X8, M));
			a5 = *(char8*)(&get2d(A, Y8 + 5, X8, M));
			a6 = *(char8*)(&get2d(A, Y8 + 6, X8, M));
			a7 = *(char8*)(&get2d(A, Y8 + 7, X8, M));

			//inner transpose: 8 * 8 = 64
			*(char8*)(&get2d(AT, X8    , Y8, N)) = char8{ a0.x0, a1.x0, a2.x0, a3.x0, a4.x0, a5.x0, a6.x0, a7.x0 };
			*(char8*)(&get2d(AT, X8 + 1, Y8, N)) = char8{ a0.y0, a1.y0, a2.y0, a3.y0, a4.y0, a5.y0, a6.y0, a7.y0 };
			*(char8*)(&get2d(AT, X8 + 2, Y8, N)) = char8{ a0.z0, a1.z0, a2.z0, a3.z0, a4.z0, a5.z0, a6.z0, a7.z0 };
			*(char8*)(&get2d(AT, X8 + 3, Y8, N)) = char8{ a0.w0, a1.w0, a2.w0, a3.w0, a4.w0, a5.w0, a6.w0, a7.w0 };
			*(char8*)(&get2d(AT, X8 + 4, Y8, N)) = char8{ a0.x1, a1.x1, a2.x1, a3.x1, a4.x1, a5.x1, a6.x1, a7.x1 };
			*(char8*)(&get2d(AT, X8 + 5, Y8, N)) = char8{ a0.y1, a1.y1, a2.y1, a3.y1, a4.y1, a5.y1, a6.y1, a7.y1 };
			*(char8*)(&get2d(AT, X8 + 6, Y8, N)) = char8{ a0.z1, a1.z1, a2.z1, a3.z1, a4.z1, a5.z1, a6.z1, a7.z1 };
			*(char8*)(&get2d(AT, X8 + 7, Y8, N)) = char8{ a0.w1, a1.w1, a2.w1, a3.w1, a4.w1, a5.w1, a6.w1, a7.w1 };
		}
		A += strideA; AT += strideA;
	}
}

#endif


//N % 4 == 0, M % 4 == 0
#ifndef IMG_BATCH_MAT_TRANSPOSE_KERNEL_4_4
#define IMG_BATCH_MAT_TRANSPOSE_KERNEL_4_4

//Size = 32, Time = 0.088 mesc, Speed = 355.114 GB/s
__global__ void img_batch_mat_transpose_kernel_4_4(
	const char* __restrict__ A,
	      char* __restrict__ AT,
	int Batch, int N, int M)
{
	int B = (blockIdx.z*blockDim.z) + threadIdx.z;
	int Y = (blockIdx.y*blockDim.y) + threadIdx.y;
	int X = (blockIdx.x*blockDim.x) + threadIdx.x;
	int Boffset = B * N * M; A += Boffset; AT += Boffset;

	int stepB = (blockDim.z*gridDim.z), strideA = N * M * stepB;
	int stepY4 = (blockDim.y*gridDim.y) << 2;
	int stepX4 = (blockDim.x*gridDim.x) << 2;

	char4 a0, a1, a2, a3;
	for (; B < Batch; B += stepB)//mul(dim[0] -> dim[-3])
	{
		for (int Y4 = (Y << 2); Y4 < N; Y4 += stepY4)
		for (int X4 = (X << 2); X4 < M; X4 += stepX4)
		{
			a0 = *(char4*)(&get2d(A, Y4    , X4, M));
			a1 = *(char4*)(&get2d(A, Y4 + 1, X4, M));
			a2 = *(char4*)(&get2d(A, Y4 + 2, X4, M));
			a3 = *(char4*)(&get2d(A, Y4 + 3, X4, M));

			//inner transpose: 4 * 4 = 16
			*(char4*)(&get2d(AT, X4    , Y4, N)) = char4{ a0.x, a1.x, a2.x, a3.x };
			*(char4*)(&get2d(AT, X4 + 1, Y4, N)) = char4{ a0.y, a1.y, a2.y, a3.y };
			*(char4*)(&get2d(AT, X4 + 2, Y4, N)) = char4{ a0.z, a1.z, a2.z, a3.z };
			*(char4*)(&get2d(AT, X4 + 3, Y4, N)) = char4{ a0.w, a1.w, a2.w, a3.w };
		}
		A += strideA; AT += strideA;
	}
}

#endif


//N % 2 == 0, M % 2 == 0
#ifndef IMG_BATCH_MAT_TRANSPOSE_KERNEL_2_2
#define IMG_BATCH_MAT_TRANSPOSE_KERNEL_2_2

//Size = 32, Time = 0.093 mesc, Speed = 336.021 GB/s
__global__ void img_batch_mat_transpose_kernel_2_2(
	const char* __restrict__ A,
	      char* __restrict__ AT,
	int Batch, int N, int M)
{
	int B = (blockIdx.z*blockDim.z) + threadIdx.z;
	int Y = (blockIdx.y*blockDim.y) + threadIdx.y;
	int X = (blockIdx.x*blockDim.x) + threadIdx.x;
	int Boffset = B * N * M; A += Boffset; AT += Boffset;

	int stepB = (blockDim.z*gridDim.z), strideA = N * M * stepB;
	int stepY2 = (blockDim.y*gridDim.y) << 1;
	int stepX2 = (blockDim.x*gridDim.x) << 1;

	char2 a0, a1;
	for (; B < Batch; B += stepB)//mul(dim[0] -> dim[-3])
	{
		for (int Y2 = (Y << 1); Y2 < N; Y2 += stepY2)
		for (int X2 = (X << 1); X2 < M; X2 += stepX2)
		{
			a0 = *(char2*)(&get2d(A, Y2    , X2, M));
			a1 = *(char2*)(&get2d(A, Y2 + 1, X2, M));

			//inner transpose: 2 * 2 = 4
			*(char2*)(&get2d(AT, X2    , Y2, N)) = char2{ a0.x, a1.x };
			*(char2*)(&get2d(AT, X2 + 1, Y2, N)) = char2{ a0.y, a1.y };
		}
		A += strideA; AT += strideA;
	}
}

#endif


//Integration
#ifndef IMG_BATCH_MAT_TRANSPOSE_FUNCTION
#define IMG_BATCH_MAT_TRANSPOSE_FUNCTION

bool __img_batch_mat_transpose(cudaStream_t stream,
	const char* __restrict__ A,
	      char* __restrict__ AT,
	int Batch, int N, int M)
{
	if (!(N & 7) && !(M & 7)) {//[LBY, LBX, LTY, LTX], { LTY, LTX } >= 3
		if (N >  63 && M >  31) { img_batch_matTrans_k88(stream, 3, 2, 3, 3, A, AT, Batch, N, M); return true; }//(6, 5), LB = 6
		if (N >  31 && M >  63) { img_batch_matTrans_k88(stream, 2, 3, 3, 3, A, AT, Batch, N, M); return true; }//(5, 6), LB = 6
		if (N > 127 && M >  15) { img_batch_matTrans_k88(stream, 4, 1, 3, 3, A, AT, Batch, N, M); return true; }//(7, 4), LB = 6
		if (N >  15 && M > 127) { img_batch_matTrans_k88(stream, 1, 4, 3, 3, A, AT, Batch, N, M); return true; }//(4, 7), LB = 6
		if (N > 255 && M >   7) { img_batch_matTrans_k88(stream, 5, 0, 3, 3, A, AT, Batch, N, M); return true; }//(8, 0), LB = 6
		if (N >   7 && M > 255) { img_batch_matTrans_k88(stream, 0, 5, 3, 3, A, AT, Batch, N, M); return true; }//(0, 8), LB = 6
	}

	if (!(N & 3) && !(M & 3)) {//[LBY, LBX, LTY, LTX], { LTY, LTX } >= 2
		if (N > 31 && M > 15) { img_batch_matTrans_k44(stream, 3, 2, 2, 2, A, AT, Batch, N, M); return true; }//(5, 4), LB = 5
		if (N > 15 && M > 31) { img_batch_matTrans_k44(stream, 2, 3, 2, 2, A, AT, Batch, N, M); return true; }//(4, 5), LB = 5
		if (N > 63 && M >  7) { img_batch_matTrans_k44(stream, 4, 1, 2, 2, A, AT, Batch, N, M); return true; }//(6, 3), LB = 5
		if (N >  7 && M > 63) { img_batch_matTrans_k44(stream, 1, 4, 2, 2, A, AT, Batch, N, M); return true; }//(3, 6), LB = 5
		if (N > 127         ) { img_batch_matTrans_k44(stream, 5, 0, 2, 2, A, AT, Batch, N, M); return true; }//(7, 2), LB = 5
		if (M > 127         ) { img_batch_matTrans_k44(stream, 0, 5, 2, 2, A, AT, Batch, N, M); return true; }//(2, 7), LB = 5
		if (N > 15 && M > 15) { img_batch_matTrans_k44(stream, 2, 2, 2, 2, A, AT, Batch, N, M); return true; }//(4, 4), LB = 4
		if (N > 15 && M > 15) { img_batch_matTrans_k44(stream, 2, 2, 2, 2, A, AT, Batch, N, M); return true; }//(4, 4), LB = 4
		if (N > 31 && M >  7) { img_batch_matTrans_k44(stream, 3, 1, 2, 2, A, AT, Batch, N, M); return true; }//(5, 3), LB = 4
		if (N >  7 && M > 31) { img_batch_matTrans_k44(stream, 1, 3, 2, 2, A, AT, Batch, N, M); return true; }//(3, 5), LB = 4
		if (N > 63          ) { img_batch_matTrans_k44(stream, 4, 0, 2, 2, A, AT, Batch, N, M); return true; }//(7, 2), LB = 4
		if (M > 63          ) { img_batch_matTrans_k44(stream, 0, 4, 2, 2, A, AT, Batch, N, M); return true; }//(2, 7), LB = 4
	}

	if (!(N & 1) && !(M & 1)) {//[LBY, LBX, LTY, LTX], { LTY, LTX } >= 1
		if (N > 15 && M >  7) { img_batch_matTrans_k22(stream, 3, 2, 1, 1, A, AT, Batch, N, M); return true; }//(4, 3), LB = 5
		if (N >  7 && M > 15) { img_batch_matTrans_k22(stream, 2, 3, 1, 1, A, AT, Batch, N, M); return true; }//(3, 4), LB = 5 
		if (N > 31          ) { img_batch_matTrans_k22(stream, 4, 1, 1, 1, A, AT, Batch, N, M); return true; }//(5, 2), LB = 5
		if (M > 31          ) { img_batch_matTrans_k22(stream, 1, 4, 1, 1, A, AT, Batch, N, M); return true; }//(2, 5), LB = 5 
		if (N >  7 && M >  7) { img_batch_matTrans_k22(stream, 2, 2, 1, 1, A, AT, Batch, N, M); return true; }//(3, 3), LB = 4
		if (N > 15          ) { img_batch_matTrans_k22(stream, 3, 1, 1, 1, A, AT, Batch, N, M); return true; }//(4, 2), LB = 4 
		if (M > 15          ) { img_batch_matTrans_k22(stream, 1, 3, 1, 1, A, AT, Batch, N, M); return true; }//(2, 4), LB = 4
	}
	return false;
}

#endif


//======[when Batch = 1, let: gridDim.z = blockDim.z = 1]=============
#ifndef IMG_MAT_TRANSPOSE_KERNEL_CALL
#define IMG_MAT_TRANSPOSE_KERNEL_CALL

//LTY>>3, LTX>>3
#define img_matTrans_k88(stream, LBY, LBX, LTY, LTX, A, AT, N, M) \
	img_batch_mat_transpose_kernel_8_8\
		<<< dim3((M>>LBX>>LTX), (N>>LBY>>LTY), 1),\
			 dim3(1<<LBX, 1<<LBY, 1), 0, stream >>>\
				(A, AT, 1, N, M)

//LTY>>2, LTX>>2
#define img_matTrans_k44(stream, LBY, LBX, LTY, LTX, A, AT, N, M) \
	img_batch_mat_transpose_kernel_4_4\
		<<< dim3((M>>LBX>>LTX), (N>>LBY>>LTY), 1),\
			 dim3(1<<LBX, 1<<LBY, 1), 0, stream >>>\
				(A, AT, 1, N, M)

//LTY>>1, LTX>>1
#define img_matTrans_k22(stream, LBY, LBX, LTY, LTX, A, AT, N, M) \
	img_batch_mat_transpose_kernel_2_2\
		<<< dim3((M>>LBX>>LTX), (N>>LBY>>LTY), 1),\
		    dim3(1<<LBX, 1<<LBY, 1), 0, stream >>>\
				(A, AT, 1, N, M)

#endif


#ifndef IMG_MAT_TRANSPOSE_FUNCTION
#define IMG_MAT_TRANSPOSE_FUNCTION

bool __img_mat_transpose(cudaStream_t stream,
	const char* __restrict__ A,
	      char* __restrict__ AT,
	int N, int M)
{
	if (!(N & 7) && !(M & 7)) {//[LBY, LBX, LTY, LTX], { LTY, LTX } >= 3
		if (N >  63 && M >  31) { img_matTrans_k88(stream, 3, 2, 3, 3, A, AT, N, M); return true; }//(6, 5), LB = 6
		if (N >  31 && M >  63) { img_matTrans_k88(stream, 2, 3, 3, 3, A, AT, N, M); return true; }//(5, 6), LB = 6
		if (N > 127 && M >  15) { img_matTrans_k88(stream, 4, 1, 3, 3, A, AT, N, M); return true; }//(7, 4), LB = 6
		if (N >  15 && M > 127) { img_matTrans_k88(stream, 1, 4, 3, 3, A, AT, N, M); return true; }//(4, 7), LB = 6
		if (N > 255 && M >   7) { img_matTrans_k88(stream, 5, 0, 3, 3, A, AT, N, M); return true; }//(8, 0), LB = 6
		if (N >   7 && M > 255) { img_matTrans_k88(stream, 0, 5, 3, 3, A, AT, N, M); return true; }//(0, 8), LB = 6
	}

	if (!(N & 3) && !(M & 3)) {//[LBY, LBX, LTY, LTX], { LTY, LTX } >= 2
		if (N > 31 && M > 15) { img_matTrans_k44(stream, 3, 2, 2, 2, A, AT, N, M); return true; }//(5, 4), LB = 5
		if (N > 15 && M > 31) { img_matTrans_k44(stream, 2, 3, 2, 2, A, AT, N, M); return true; }//(4, 5), LB = 5
		if (N > 63 && M >  7) { img_matTrans_k44(stream, 4, 1, 2, 2, A, AT, N, M); return true; }//(6, 3), LB = 5
		if (N >  7 && M > 63) { img_matTrans_k44(stream, 1, 4, 2, 2, A, AT, N, M); return true; }//(3, 6), LB = 5
		if (N > 127         ) { img_matTrans_k44(stream, 5, 0, 2, 2, A, AT, N, M); return true; }//(7, 2), LB = 5
		if (M > 127         ) { img_matTrans_k44(stream, 0, 5, 2, 2, A, AT, N, M); return true; }//(2, 7), LB = 5
		if (N > 15 && M > 15) { img_matTrans_k44(stream, 2, 2, 2, 2, A, AT, N, M); return true; }//(4, 4), LB = 4
		if (N > 15 && M > 15) { img_matTrans_k44(stream, 2, 2, 2, 2, A, AT, N, M); return true; }//(4, 4), LB = 4
		if (N > 31 && M >  7) { img_matTrans_k44(stream, 3, 1, 2, 2, A, AT, N, M); return true; }//(5, 3), LB = 4
		if (N >  7 && M > 31) { img_matTrans_k44(stream, 1, 3, 2, 2, A, AT, N, M); return true; }//(3, 5), LB = 4
		if (N > 63          ) { img_matTrans_k44(stream, 4, 0, 2, 2, A, AT, N, M); return true; }//(7, 2), LB = 4
		if (M > 63          ) { img_matTrans_k44(stream, 0, 4, 2, 2, A, AT, N, M); return true; }//(2, 7), LB = 4
	}

	if (!(N & 1) && !(M & 1)) {//[LBY, LBX, LTY, LTX], { LTY, LTX } >= 1
		if (N > 15 && M >  7) { img_matTrans_k22(stream, 3, 2, 1, 1, A, AT, N, M); return true; }//(4, 3), LB = 5
		if (N >  7 && M > 15) { img_matTrans_k22(stream, 2, 3, 1, 1, A, AT, N, M); return true; }//(3, 4), LB = 5 
		if (N > 31          ) { img_matTrans_k22(stream, 4, 1, 1, 1, A, AT, N, M); return true; }//(5, 2), LB = 5
		if (M > 31          ) { img_matTrans_k22(stream, 1, 4, 1, 1, A, AT, N, M); return true; }//(2, 5), LB = 5 
		if (N >  7 && M >  7) { img_matTrans_k22(stream, 2, 2, 1, 1, A, AT, N, M); return true; }//(3, 3), LB = 4
		if (N > 15          ) { img_matTrans_k22(stream, 3, 1, 1, 1, A, AT, N, M); return true; }//(4, 2), LB = 4 
		if (M > 15          ) { img_matTrans_k22(stream, 1, 3, 1, 1, A, AT, N, M); return true; }//(2, 4), LB = 4
	}
	return false;
}

#endif

#endif