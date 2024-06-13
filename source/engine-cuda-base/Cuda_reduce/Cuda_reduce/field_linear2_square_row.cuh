#pragma once

#ifndef FIELD_LINEAR2_SQUARE_ROW_H
#define FIELD_LINEAR2_SQUARE_ROW_H

//(1) field_num % A.width ==0
//(2) field_num -> field_stride = field_num / width * stride
//(3) A.length % feature_num == 0
//(4) A.lengthv % fieled_stride == 0
//(5) row_num = A.lengthv / field_stride
//(6) [y, x] -> [row_num, field_stride]
//(7) [N, M] = [row_num, field_stride], A.lengthv = N*M
#ifndef FIELD_LINEAR2_SQUARE_ROW_CALL
#define FIELD_LINEAR2_SQUARE_ROW_CALL

//[16, 4]
#define field_linear2_square_row4_small(stream, X1, X2, C, alpha, beta, gamma, N, M, V, width, stride) \
	field_linear2_square_row_kernel_4<3, 3>\
		<<< dim3(MIN_int32((M + 31)>>5, GRID_MAX),\
                 MIN_int32((N + 63)>>6, GRID_MAX)),\
            dim3(8, 8), 0, stream>>>\
			(X1, X2, C, alpha, beta, gamma, N, M, V, width, stride)

//LTX >=2
#define field_linear2_square_row4(stream, LBY, LBX, LTY, LTX, X1, X2, C, alpha, beta, gamma, N, M, V, width, stride) \
	field_linear2_square_row_kernel_4<LBY, LBX>\
		<<< dim3(MIN_int32(M>>LBX>>LTX, GRID_MAX),\
                 MIN_int32(N>>LBY>>LTY, GRID_MAX)),\
            dim3(1<<LBX, 1<<LBY), 0, stream >>>\
			(X1, X2, C, alpha, beta, gamma, N, M, V, width, stride)

#endif


#ifndef FIELD_LINEAR2_SQUARE_ROW_KERNEL_4
#define FIELD_LINEAR2_SQUARE_ROW_KERNEL_4

//sum: C * (alpha*X1 + beta*X2 + gamma)^2
template<int LBY, int LBX>
__global__ void field_linear2_square_row_kernel_4(
	const float* __restrict__ X1,
	const float* __restrict__ X2, 
	float C, float alpha, float beta, float gamma,
	int N, int M,
	float* __restrict__ V,
	int width, int stride)
{
	const int by = blockIdx.y, bx = blockIdx.x;
	const int ty = threadIdx.y, tx = threadIdx.x;

	__shared__ float2 As[1 << LBX][(2 << LBY) + 2];

	//------[parallel field num = 4]----------------------------------------
	const int offsetY = (by << LBY) + ty, stepY = (gridDim.y << LBY);
	const int offsetX = (bx << LBX) + tx, stepX = (gridDim.x << LBX);
	const int stepX4 = stepX << 2, M4 = (M >> 2) << 2;
	for (int x4 = offsetX << 2; x4 < M4; x4 += stepX4)
	{
		//------compute area[thread reduce: 4 local result]-----------------
		float4 x2 = *(float4*)(X2 + x4);
		float4 c = F32_4_0;//solve the errors
		float4 v = F32_4_0;//thread reduce: 4 local result
		for (int y = offsetY; y < N; y += stepY) 
		{
			const int xoffset = y * M + x4;//X1[y, x4]
			float4 x1 = *(float4*)(X1 + xoffset);

			float4 a;//V = C * (alpha*X1 + beta*X2 + gamma)^2
			a.x = alpha * x1.x + beta * x2.x + gamma; a.x = a.x * a.x;
			a.y = alpha * x1.y + beta * x2.y + gamma; a.y = a.y * a.y;
			a.z = alpha * x1.z + beta * x2.z + gamma; a.z = a.z * a.z;
			a.w = alpha * x1.w + beta * x2.w + gamma; a.w = a.w * a.w;
			Kahan_simdAdd4(v, a, c);
		}
		v.x = v.x * C; c.x = c.x * C;
		v.y = v.y * C; c.y = c.y * C;
		v.z = v.z * C; c.z = c.z * C;
		v.w = v.w * C; c.w = c.w * C;
		*(float4*)(&As[tx][ty << 1]) = v;
		__syncthreads();

		//------compute area[block reduce: 4 global result]-----------------
		int yIdx;
		float2 sc[2]{//with same tx: overlap the error in same field 
			float2{ c.x, c.y },//yIdx % 2 == 0
			float2{ c.z, c.w } //yIdx % 2 != 0
		};

		if (LBY >= 6) {
			yIdx = ((ty & 31) << 1) + (ty >> 5);
			if (ty < 64) {//128 -> 64
				float2 v1 = As[tx][yIdx], v2 = As[tx][yIdx + 64], c = sc[(yIdx & 1)];
				Kahan_simdAdd2(v1, v2, c); As[tx][yIdx] = v1;
			}
			__syncthreads();
		}
		if (LBY >= 5) {
			yIdx = ((ty & 15) << 1) + (ty >> 4);
			if (ty < 32) {//64 -> 32
				float2 v1 = As[tx][yIdx], v2 = As[tx][yIdx + 32], c = sc[(yIdx & 1)];
				Kahan_simdAdd2(v1, v2, c); As[tx][yIdx] = v1;
			}
			__syncthreads();
		}
		if (LBY >= 4) {
			yIdx = ((ty & 7) << 1) + (ty >> 3);
			if (ty < 16) {//32 -> 16
				float2 v1 = As[tx][yIdx], v2 = As[tx][yIdx + 16], c = sc[(yIdx & 1)];
				Kahan_simdAdd2(v1, v2, c); As[tx][yIdx] = v1;
			}
			__syncthreads();
		}

		yIdx = ((ty & 3) << 1) + (ty >> 2);
		if (ty < 8) {//16 -> 8
			float2 v1 = As[tx][yIdx], v2 = As[tx][yIdx + 8], c = sc[(yIdx & 1)];
			Kahan_simdAdd2(v1, v2, c); As[tx][yIdx] = v1;
		}
		__syncthreads();

		yIdx = ((ty & 1) << 1) + (ty >> 1);
		if (ty < 4) {//8 -> 4
			float2 v1 = As[tx][yIdx], v2 = As[tx][yIdx + 4], c = sc[(yIdx & 1)];
			Kahan_simdAdd2(v1, v2, c); As[tx][yIdx] = v1;
		}
		__syncthreads();

		yIdx = ((ty & 0) << 1) + ty;
		if (ty < 2) {//4 -> 2, save
			float2 v1 = As[tx][yIdx], v2 = As[tx][yIdx + 2], c = sc[(yIdx & 1)];
			Kahan_simdAdd2(v1, v2, c);

			const int xindex2 = x4 + (ty << 1);
			within_width2(v1, xindex2, stride, width);
			*(float2*)(&get(V, by, xindex2, M)) = v1;
		}
		__syncthreads();
	}
}

#endif


//======[integration]========================================
#ifndef FIELD_LINEAR2_SQUARE_ROW_STAGE
#define FIELD_LINEAR2_SQUARE_ROW_STAGE

//M % 4 == 0, M >= 4
void __field_linear2_square_row_stage(cudaStream_t stream,
	const float* X1,
	const float* X2,
	float C, float alpha, float beta, float gamma,
	int N, int M,
	float* V, int width, int stride)
{
	if (M > 15) {
		if (N > 63) { field_linear2_square_row4(stream, 3, 2, 3, 2, X1, X2, C, alpha, beta, gamma, N, M, V, width, stride); return; }//[64, 16]
		if (N > 31) { field_linear2_square_row4(stream, 3, 2, 2, 2, X1, X2, C, alpha, beta, gamma, N, M, V, width, stride); return; }//[32, 16]
		if (N > 15) { field_linear2_square_row4(stream, 3, 2, 1, 2, X1, X2, C, alpha, beta, gamma, N, M, V, width, stride); return; }//[16, 16]
	}
	if (M > 7) {
		if (N > 127) { field_linear2_square_row4(stream, 4, 1, 3, 2, X1, X2, C, alpha, beta, gamma, N, M, V, width, stride); return; }//[ 64, 8]
		if (N >  63) { field_linear2_square_row4(stream, 4, 1, 2, 2, X1, X2, C, alpha, beta, gamma, N, M, V, width, stride); return; }//[ 32, 8]
		if (N >  31) { field_linear2_square_row4(stream, 4, 1, 1, 2, X1, X2, C, alpha, beta, gamma, N, M, V, width, stride); return; }//[ 16, 8]
	}
	field_linear2_square_row4_small(stream, X1, X2, C, alpha, beta, gamma, N, M, V, width, stride);
}

#endif 


#ifndef FIELD_LINEAR2_SQUARE_ROW
#define FIELD_LINEAR2_SQUARE_ROW

int __field_linear2_square_row(cudaStream_t stream,
	const float* X1, 
	const float* X2,
	float C, float alpha, float beta, float gamma,
	int N, int M,
	float* V, float *Y,
	int width, int stride,
	int partNum)
{
	int nextN = field_nextN(N, M);//N <= HV
	if (nextN <= partNum) {//V.address = NULL, only 1 stage, write result to Y
		__field_linear2_square_row_stage(stream, X1, X2, C, alpha, beta, gamma, N, M, Y, width, stride);
		return nextN;
	}

	//V.address != NULL, at least 2 stages
	__field_linear2_square_row_stage(stream, X1, X2, C, alpha, beta, gamma, N, M, V, width, stride);

	N = nextN, nextN = field_nextN(N, M);
	while (nextN > partNum)//end: nextN <= partNum
	{
		__field_sum_stage(stream, V, N, M, V, width, stride);
		N = nextN, nextN = field_nextN(nextN, M);
	}
	__field_sum_stage(stream, V, N, M, Y, width, stride);//the last stage
	return nextN;
}

#endif

#endif