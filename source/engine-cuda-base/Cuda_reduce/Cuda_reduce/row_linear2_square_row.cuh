#pragma once

#ifndef ROW_LINEAR2_SQUARE_ROW_H
#define ROW_LINEAR2_SQUARE_ROW_H

//M = row_lengthv, N = field_lengthv
//(1) field_num % [A.width, B.width] ==0
//(2) field_num -> field_stride = field_num / width * stride
//(3) [A.length,  B.length ] % feature_num == 0
//(4) [A.lengthv, B.lengthv] % fieled_stride == 0
//(5) row_num = A.lengthv / field_stride
//(6) [y, x] -> [row_num, field_stride]
//(7) [N, M] = [row_num, field_stride], A.lengthv = N*M
#ifndef ROW_LINEAR2_SQUARE_ROW_CALL
#define ROW_LINEAR2_SQUARE_ROW_CALL

//LBX>=4
#define row_linear2_square_row_fast16(stream, LBY, LBX, LTY, LTX, X1, X2, C, alpha, beta, gamma, N, M, V, SV, width, stride) \
	row_linear2_square_row_kernel_fast_16<LBY, LBX>\
		<<< dim3(MIN_int32((M >> LBX >> LTX), GRID_MAX),\
                 MIN_int32(((N + (1<<LBY<<LTY) - 1)>>LBY>>LTY), GRID_MAX)),\
			dim3(1<<LBX, 1<<LBY), 0, stream >>>\
			(X1, X2, C, alpha, beta, gamma, N, M, V, SV, width, stride)

//LBX>=3
#define row_linear2_square_row_fast8(stream, LBY, LBX, LTY, LTX, X1, X2, C, alpha, beta, gamma, N, M, V, SV, width, stride) \
	row_linear2_square_row_kernel_fast_8<LBY, LBX>\
		<<< dim3(MIN_int32((M >> LBX >> LTX), GRID_MAX),\
                 MIN_int32(((N + (1<<LBY<<LTY) - 1)>>LBY>>LTY), GRID_MAX)),\
			dim3(1<<LBX, 1<<LBY), 0, stream >>> \
			(X1, X2, C, alpha, beta, gamma, N, M, V, SV, width, stride)

//LBX=5, LTX=0, used for: M<=32
#define row_linear2_square_row_small(stream, LBY, LTY, X1, X2, C, alpha, beta, gamma, N, M, V, SV, width, stride) \
	row_linear2_square_row_kernel_slow_16<LBY, 5>\
		<<< dim3(1, MIN_int32(((N + (1<<LBY<<LTY) - 1)>>LBY>>LTY), GRID_MAX)),\
		    dim3(32, 1<<LBY), 0, stream >>>\
			(X1, X2, C, alpha, beta, gamma, N, M, V, SV, width, stride)

#endif


#ifndef ROW_LINEAR2_SQUARE_ROW_FAST_16
#define ROW_LINEAR2_SQUARE_ROW_FAST_16

template<int LBY, int LBX>
__global__ void row_linear2_square_row_kernel_fast_16(
	const float* __restrict__ X1,
	const float* __restrict__ X2,
	float C, float alpha, float beta, float gamma,
	int N, int M,
	float* __restrict__ V, int SV,//stride of V
	int width, int stride)
{
	const int by = blockIdx.y, bx = blockIdx.x;
	const int ty = threadIdx.y, tx = threadIdx.x;

	__shared__ float As[1 << LBY][(1 << LBX) + 1];

	int y = (by << LBY) + ty, Xoffset = ((bx << LBX) + tx) << 2;
	const int stepY = (gridDim.y << LBY);//go to the next row for this thread
	const int stepX4 = (gridDim.x << LBX) << 2;//go to the next element of the current row
	const int Yoffset = y * M, strideX = stepY * M;

	for (X1 += Yoffset; y < N; y += stepY)
	{
		float b = beta * X2[y] + gamma;
		float c = 0.0f;//solve the error
		float v = 0.0f;//thread local reduce
		for (int x = Xoffset; x < M; x += stepX4)
		{
			float4 x1 = *(float4*)(X1 + x);

			float4 a;//V = C * (alpha*X1 + beta*X2 + gamma)^2
			a.x = alpha * x1.x + b; a.x = a.x * a.x;
			a.y = alpha * x1.y + b; a.y = a.y * a.y;
			a.z = alpha * x1.z + b; a.z = a.z * a.z;
			a.w = alpha * x1.w + b; a.w = a.w * a.w;

			within_width4(a, x, stride, width);
			Kahan_sum4(v, a, c);//v += a.x + a.y + a.z + a.w;
		}
		v = v * C; c = c * C;
		As[ty][tx] = v;
		X1 += strideX;
		__syncthreads();

		//block global reduce[no need to zero c, as the same row]
		if (LBX >= 7) {//BLOCK_SIZE = 128
			if (tx < 64) { v = As[ty][tx]; Kahan_sum1(v, As[ty][tx + 64], c); As[ty][tx] = v; }
			__syncthreads();
		}
		if (LBX >= 6) {//BLOCK_SIZE = 64
			if (tx < 32) { v = As[ty][tx]; Kahan_sum1(v, As[ty][tx + 32], c); As[ty][tx] = v; }
			__syncthreads();
		}
		if (LBX >= 5) {//BLOCK_SIZE = 32
			if (tx < 16) { v = As[ty][tx]; Kahan_sum1(v, As[ty][tx + 16], c); As[ty][tx] = v; }
			__syncthreads();
		}

		//within a warp(BLOCK_SIZE >= 16)============================
		float a; v = As[ty][tx];
		a = __shfl_down_sync(0xffffffff, v, 8); Kahan_sum1(v, a, c);
		a = __shfl_down_sync(0xffffffff, v, 4); Kahan_sum1(v, a, c);
		a = __shfl_down_sync(0xffffffff, v, 2); Kahan_sum1(v, a, c);
		a = __shfl_down_sync(0xffffffff, v, 1); Kahan_sum1(v, a, c);

		if (tx == 0) get(V, bx, y, SV) = v;//transposed: [y, bx] -> [bx, y]
		__syncthreads();
	}
}

#endif


#ifndef ROW_LINEAR2_SQUARE_ROW_FAST_8
#define ROW_LINEAR2_SQUARE_ROW_FAST_8

template<int LBY, int LBX>
__global__ void row_linear2_square_row_kernel_fast_8(
	const float* __restrict__ X1,
	const float* __restrict__ X2,
	float C, float alpha, float beta, float gamma,
	int N, int M,
	float* __restrict__ V, int SV,//stride of V
	int width, int stride)
{
	const int by = blockIdx.y, bx = blockIdx.x;
	const int ty = threadIdx.y, tx = threadIdx.x;

	__shared__ float As[1 << LBY][(1 << LBX) + 1];

	int y = (by << LBY) + ty, Xoffset = ((bx << LBX) + tx) << 2;
	const int stepY = (gridDim.y << LBY);//go to the next row for this thread
	const int stepX4 = (gridDim.x << LBX) << 2; //go to the next element of the current row
	const int Yoffset = y * M, strideX = stepY * M;

	for (X1 += Yoffset; y < N; y += stepY)
	{
		float b = beta * X2[y] + gamma;
		float c = 0.0f;//solve the error
		float v = 0.0f;//thread local reduce
		for (int x = Xoffset; x < M; x += stepX4)//thread local reduce
		{
			float4 x1 = *(float4*)(X1 + x);

			float4 a;//V = C * (alpha*X1 + beta*X2 + gamma)^2
			a.x = alpha * x1.x + b; a.x = a.x * a.x;
			a.y = alpha * x1.y + b; a.y = a.y * a.y;
			a.z = alpha * x1.z + b; a.z = a.z * a.z;
			a.w = alpha * x1.w + b; a.w = a.w * a.w;

			within_width4(a, x, stride, width);
			Kahan_sum4(v, a, c);//v += a.x + a.y + a.z + a.w;
		}
		v = v * C; c *= c * C; 
		As[ty][tx] = v;
		X1 += strideX;
		__syncthreads();

		//block global reduce
		if (LBX >= 7) {//BLOCK_SIZE = 128
			if (tx < 64) { v = As[ty][tx]; Kahan_sum1(v, As[ty][tx + 64], c); As[ty][tx] = v; }
			__syncthreads();
		}
		if (LBX >= 6) {//BLOCK_SIZE = 64
			if (tx < 32) { v = As[ty][tx]; Kahan_sum1(v, As[ty][tx + 32], c); As[ty][tx] = v; }
			__syncthreads();
		}
		if (LBX >= 5) {//BLOCK_SIZE = 32
			if (tx < 16) { v = As[ty][tx]; Kahan_sum1(v, As[ty][tx + 16], c); As[ty][tx] = v; }
			__syncthreads();
		}
		if (LBX >= 4) {//BLOCK_SIZE = 16
			if (tx < 8) { v = As[ty][tx]; Kahan_sum1(v, As[ty][tx + 8], c); As[ty][tx] = v; }
			__syncthreads();
		}

		//within a warp(BLOCK_SIZE >= 16)============================
		float a; v = As[ty][tx];
		a = __shfl_down_sync(0xffffffff, v, 4); Kahan_sum1(v, a, c);
		a = __shfl_down_sync(0xffffffff, v, 2); Kahan_sum1(v, a, c);
		a = __shfl_down_sync(0xffffffff, v, 1); Kahan_sum1(v, a, c);

		if (tx == 0) get(V, bx, y, SV) = v;//transposed: [y, bx] -> [bx, y]
		__syncthreads();
	}
}

#endif


#ifndef ROW_LINEAR2_SQUARE_ROW_SLOW
#define ROW_LINEAR2_SQUARE_ROW_SLOW

template<int LBY, int LBX>
__global__ void row_linear2_square_row_kernel_slow_16(
	const float* __restrict__ X1,
	const float* __restrict__ X2,
	float C, float alpha, float beta, float gamma,
	int N, int M,
	float* __restrict__ V, int SV,//stride of V
	int width, int stride)
{
	const int by = blockIdx.y, bx = blockIdx.x;
	const int ty = threadIdx.y, tx = threadIdx.x;

	__shared__ float As[1 << LBY << LBX];//(1<<LBY, 1<<LBX)
	int As_yx = (ty << LBX) + tx;//(ty, tx)

	int y = (by << LBY) + ty, Xoffset = (bx << LBX) + tx;
	const int stepY = (gridDim.y << LBY), stepX = (gridDim.x << LBX);
	const int Yoffset = y * M, strideX = stepY * M;

	for (X1 += Yoffset; y < N; y += stepY)
	{
		float b = beta * X2[y] + gamma;
		float c = 0.0f;//solve the error
		float v = 0.0f;
		for (int x = Xoffset; x < M; x += stepX)//thread local reduce
		{
			float x1 = X1[x];

			//V = C * (alpha*X1 + beta*X2 + gamma)^2
			float y = alpha * x1 + b; y = y * y;
			y *= ((x % stride) < width);//within_width

			float dv = y - c;//Kahan Add
			float t = v + dv;
			c = (t - v) - dv;
			v = t;//Kahan Add: v += a
		}
		v = v * C; c = c * C;
		As[As_yx] = v;
		X1 += strideX;
		__syncthreads();

		//block global reduce
		if (LBX >= 7) {
			if (tx < 64) As[As_yx] += As[As_yx + 64];
			__syncthreads();
		}
		if (LBX >= 6) {
			if (tx < 32) As[As_yx] += As[As_yx + 32];
			__syncthreads();
		}
		if (LBX >= 5) {
			if (tx < 16) As[As_yx] += As[As_yx + 16];
			__syncthreads();
		}
		if (tx < 8) warp_sum_8(As, As_yx);
		if (tx == 0) get(V, bx, y, SV) = As[As_yx];
		__syncthreads();
	}
}

#endif


//======[integration]========================================
#ifndef ROW_LINEAR2_SQUARE_ROW_STAGE
#define ROW_LINEAR2_SQUARE_ROW_STAGE

//HV = nextRowReduceStageHeight(M)
void __row_linear2_square_row_stage(cudaStream_t stream,
	const float* X1, const float* X2,
	float C, float alpha, float beta, float gamma,
	int N, int M,
	float* V, int SV,//stride of V
	int width, int stride)
{
	if (M > 255) {
		if (N > 31) { row_linear2_square_row_fast16(stream, 1, 4, 4, 4, X1, X2, C, alpha, beta, gamma, N, M, V, SV, width, stride); return; }
		if (N > 15) { row_linear2_square_row_fast16(stream, 1, 4, 3, 4, X1, X2, C, alpha, beta, gamma, N, M, V, SV, width, stride); return; }
		if (N >  7) { row_linear2_square_row_fast16(stream, 1, 4, 2, 4, X1, X2, C, alpha, beta, gamma, N, M, V, SV, width, stride); return; }
		if (N >  3) { row_linear2_square_row_fast16(stream, 1, 4, 1, 4, X1, X2, C, alpha, beta, gamma, N, M, V, SV, width, stride); return; }
		row_linear2_square_row_fast16(stream, 1, 4, 0, 4, X1, X2, C, alpha, beta, gamma, N, M, V, SV, width, stride); return;
	}

	if (M > 127) {
		if (N > 31) { row_linear2_square_row_fast8(stream, 2, 3, 3, 4, X1, X2, C, alpha, beta, gamma, N, M, V, SV, width, stride); return; }
		if (N > 15) { row_linear2_square_row_fast8(stream, 2, 3, 2, 4, X1, X2, C, alpha, beta, gamma, N, M, V, SV, width, stride); return; }
		if (N >  7) { row_linear2_square_row_fast8(stream, 2, 3, 1, 4, X1, X2, C, alpha, beta, gamma, N, M, V, SV, width, stride); return; }
		if (N >  3) { row_linear2_square_row_fast8(stream, 2, 3, 0, 4, X1, X2, C, alpha, beta, gamma, N, M, V, SV, width, stride); return; }
		row_linear2_square_row_fast8(stream, 1, 3, 0, 4, X1, X2, C, alpha, beta, gamma, N, M, V, SV, width, stride); return;
	}

	if (M > 63) {
		if (N > 31) { row_linear2_square_row_fast8(stream, 2, 3, 3, 3, X1, X2, C, alpha, beta, gamma, N, M, V, SV, width, stride); return; }
		if (N > 15) { row_linear2_square_row_fast8(stream, 2, 3, 2, 3, X1, X2, C, alpha, beta, gamma, N, M, V, SV, width, stride); return; }
		if (N >  7) { row_linear2_square_row_fast8(stream, 2, 3, 1, 3, X1, X2, C, alpha, beta, gamma, N, M, V, SV, width, stride); return; }
		if (N >  3) { row_linear2_square_row_fast8(stream, 2, 3, 0, 3, X1, X2, C, alpha, beta, gamma, N, M, V, SV, width, stride); return; }
		row_linear2_square_row_fast8(stream, 1, 3, 0, 3, X1, X2, C, alpha, beta, gamma, N, M, V, SV, width, stride); return;
	}

	if (M > 31) {//2^7, M >= 128
		if (N > 31) { row_linear2_square_row_fast8(stream, 2, 3, 3, 2, X1, X2, C, alpha, beta, gamma, N, M, V, SV, width, stride); return; }
		if (N > 15) { row_linear2_square_row_fast8(stream, 2, 3, 2, 2, X1, X2, C, alpha, beta, gamma, N, M, V, SV, width, stride); return; }
		if (N >  7) { row_linear2_square_row_fast8(stream, 2, 3, 1, 2, X1, X2, C, alpha, beta, gamma, N, M, V, SV, width, stride); return; }
		if (N >  3) { row_linear2_square_row_fast8(stream, 2, 3, 0, 2, X1, X2, C, alpha, beta, gamma, N, M, V, SV, width, stride); return; }
		row_linear2_square_row_fast8(stream, 1, 3, 0, 2, X1, X2, C, alpha, beta, gamma, N, M, V, SV, width, stride); return;
	}

	if (N > 31) { row_linear2_square_row_small(stream, 1, 4, X1, X2, C, alpha, beta, gamma, N, M, V, SV, width, stride); return; }
	if (N > 15) { row_linear2_square_row_small(stream, 1, 3, X1, X2, C, alpha, beta, gamma, N, M, V, SV, width, stride); return; }
	if (N >  7) { row_linear2_square_row_small(stream, 1, 2, X1, X2, C, alpha, beta, gamma, N, M, V, SV, width, stride); return; }
	if (N >  3) { row_linear2_square_row_small(stream, 1, 1, X1, X2, C, alpha, beta, gamma, N, M, V, SV, width, stride); return; }
	row_linear2_square_row_small(stream, 1, 0, X1, X2, C, alpha, beta, gamma, N, M, V, SV, width, stride);
}

#endif


#ifndef ROW_LINEAR2_SQUARE_ROW
#define ROW_LINEAR2_SQUARE_ROW

//new fashion
//V used a template buffer to store median value for: A -> Y
//Y: must 1D tensor, no inner zero between results
int __row_linear2_square_row(cudaStream_t stream,
	const float* X1,
	const float* X2,
	float C, float alpha, float beta, float gamma,
	int N, int M,
	float* V, float *Y,
	int width, int stride,
	int partNum)
{
	int nextM = row_nextM(M);
	if (nextM <= partNum) {//only 1 stage: directly write result to Y tightly, so SV = N
		__row_linear2_square_row_stage(stream, X1, X2, C, alpha, beta, gamma, N, M, Y, N, width, stride);
		return nextM;
	}

	//at least 2 stages: tansposed: //A[N, M] -> V[nextM, SV]
	int SV = (N + 3) >> 2 << 2;///make sure: SV >= 4, SV % 4 == 0
	__row_linear2_square_row_stage(stream, X1, X2, C, alpha, beta, gamma, N, M, V, SV, width, stride);

	M = nextM, nextM = field_nextN(M, SV);
	while (nextM > partNum)//end: nextN <= partNum
	{
		__field_sum_stage(stream, V, M, SV, V, N, SV);//width = N, stride = SV
		M = nextM, nextM = field_nextN(M, SV);
	}
	__field_sum_stage(stream, V, M, SV, Y, N, SV);
	return nextM;
}

#endif

#endif