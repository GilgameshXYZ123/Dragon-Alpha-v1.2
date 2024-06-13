#pragma once

#define IMG_ROW_LINEAR_H
#ifndef IMG_ROW_LINEAR_H
#define IMG_ROW_LINEAR_H

//(1) X[N, M] -> Y[N]
//(2) M = row_lengthv, N = field_length
//(3) X.ndim >= 3, Y.ndim = 1
//(4) M % stride == 0, stride % 4 == 0, stride >= 4
#ifndef IMG_ROW_LINEAR_CALL
#define IMG_ROW_LINEAR_CALL

//
#define fast_img_row_linear(stream, LBY, LBX, LTY, LTX, A, alpha, beta, N, M, V, SV, width, stride) \
	fast_img_row_linear_kernel<LBY, LBX>\
		<<< dim3((M >> LBX >> LTX), ((N + (1<<LBY<<LTY) - 1)>>LBY>>LTY) ),\
			dim3(1<<LBX, 1<<LBY), 0, stream >>>\
			(A, alpha, beta, N, M, V, SV, width, stride)

//common
#define naive_img_row_linear(stream, LB, LT, X, alpha, beta, N, M, Y, width, stride)\
	naive_img_row_linear_kernel\
		<<< N>>LB>>LT, (1<<LB), 0, stream >>>\
			(X, alpha, beta, N, M, Y, width, stride)

//common
#define naive_img_row_linear_small(stream, X, alpha, beta, N, M, Y, width, stride)\
	naive_img_row_linear_kernel\
		<<< 1, ((N + 3) >> 2), 0, stream >>>\
			(X, alpha, beta, N, M, Y, width, stride)

#endif


#ifndef FAST_IMG_ROW_LINEAR
#define FAST_IMG_ROW_LINEAR

//M = row_lengthv
//N = field_lengthv
template<int LBY, int LBX>
__global__ void fast_img_row_linear_kernel(
	const char* __restrict__ A,
	float alpha, float beta,
	int N, int M,
	char* __restrict__ V, int SV,//stride of V
	int width, int stride)
{
	int by = blockIdx.y, bx = blockIdx.x;
	int ty = threadIdx.y, tx = threadIdx.x;

	__shared__ float As[1 << LBY << LBX];//[1 << LBY, 1 << LBX]
	int As_yx = (ty << LBX) + tx;//(ty, tx)

	int y = (by << LBY) + ty, offsetX = ((bx << LBX) + tx) << 2;
	const int stepY = (gridDim.y << LBY);//go to the next row for this thread
	const int stepX4 = (gridDim.x << LBX) << 2; //go to the next element of the current row

	for (A += y * M; y < N; y += stepY)
	{
		float c = 0.0f;//solve the error
		float v = 0.0f;
		for (int x = offsetX; x < M; x += stepX4)//thread local reduce
		{
			uchar4 av = *(uchar4*)(A + x);
			float4 a;//V = row_sum: alpha*X[i] + beta
			a.x = alpha * av.x + beta;
			a.y = alpha * av.y + beta;
			a.z = alpha * av.z + beta;
			a.w = alpha * av.w + beta;
			within_width4(a, x, stride, width);

			float dv = (a.x + a.y + a.z + a.w) - c;
			float t = v + dv;
			c = (t - v) - dv;
			v = t;//v += a.x + a.y + a.z + a.w;
		}
		As[As_yx] = v;

		A += stepY * M;//A[Y + stepY][0]
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
		if (tx < 8) warp_sum_8(As, As_yx);//LBX >= 4
		if (tx == 0) {
			unsigned char result = PIXEL_CLIP(As[As_yx]);
			get(V, bx, y, SV) = result;
		}
		__syncthreads();
	}
}

#endif


//M <= 32:
#ifndef NAIVE_IMG_ROW_LINEAR
#define NAIVE_IMG_ROW_LINEAR

//(5, 3): Size = 32.5, Time = 0.129 mesc, Speed = 246.033 GB/s
//(5, 2): Size = 32.5, Time = 0.119 mesc, Speed = 266.708 GB/s
//(5, 1): Size = 32.5, Time = 0.119 mesc, Speed = 266.708 GB/s
__global__ void naive_img_row_linear_kernel(
	const char* __restrict__ X, 
	float alpha, float beta,
	int N, int M,
	char* __restrict__ Y,
	int width, int stride)
{
	int step = gridDim.x*blockDim.x;
	int yindex = blockIdx.x*blockDim.x + threadIdx.x;

	for (int y = yindex; y < N; y += step)
	{
		float v = 0.0f, c = 0.0f;
		const int y_stride = y * stride;
		for (int x4 = 0; x4 < M; x4 += 4) 
		{
			const int xoffset = y_stride + x4;//X[y, x4]
			uchar4 xv = *(uchar4*)(X + xoffset);
			
			float4 a;//Y = sum: alpha*X[i] + beta
			a.x = alpha * xv.x + beta;
			a.y = alpha * xv.y + beta;
			a.z = alpha * xv.z + beta;
			a.w = alpha * xv.w + beta;
			within_width4(a, x4, stride, width);

			float dv = (a.x + a.y + a.z + a.w) - c;
			float t = v + dv;
			c = (t - v) - dv;
			v = t;//v += a.x + a.y + a.z + a.w;
		}
		Y[y] = PIXEL_CLIP(v);
	}
}

#endif


void __img_row_linear2D(cudaStream_t stream,
	const char* X,
	float alpha, float beta,
	int N, int M,
	char* Y, 
	int width, int stride)
{
	if (N < 256) { naive_img_row_linear_small(stream, X, alpha, beta, N, M, Y, width, stride); return; }
	if (N >= 2048) { naive_img_row_linear(stream, 5, 2, X, alpha, beta, N, M, Y, width, stride); return; }
	naive_img_row_linear(stream, 5, 1, X, alpha, beta, N, M, Y, width, stride);
}

#endif
