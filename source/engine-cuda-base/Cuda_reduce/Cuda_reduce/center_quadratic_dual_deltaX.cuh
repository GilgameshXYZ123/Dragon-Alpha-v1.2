#pragma once

#ifndef CENTER_QUADRATIC_DUAL_DELTAX_H
#define CENTER_QUADRATIC_DUAL_DELTAX_H

//reduction: along dim1 axis
//center_reduce: A[dim0, dim1, dim2] 
//	-> field_reduce: A'[dim1, dim0*dim2] 
//	= V[HV, dim0 * dim2]
//HV = next_center(N, M), N = height, M = width
//(1) height = dim1, height % 4 != 0
//(2) width = dim0*dim2, width % 4 = 0
#ifndef CENTER_QUADRATIC_DUAL_DELTAX_CALL
#define CENTER_QUADRATIC_DUAL_DELTAX_CALL

//[16, 4]
#define center_quadratic_dual_deltaX4_small(stream, deltaY, X1, X2, k11, k12, k22, k1, k2, C, dim1, dim2, N, M, deltaX1, V, width, stride) \
	center_quadratic_dual_deltaX_kernel_4<3, 3>\
		<<< dim3(MIN_int32((M + 31)>>5, GRID_MAX),\
                 MIN_int32((N + 63)>>6, GRID_MAX)),\
            dim3(8, 8), 0, stream>>>\
			(deltaY, X1, X2, k11, k12, k22, k1, k2, C, dim1, dim2, N, M, deltaX1, V, width, stride)

//LTX >=2
#define center_quadratic_dual_deltaX4(stream, LBY, LBX, LTY, LTX, deltaY, X1, X2, k11, k12, k22, k1, k2, C, dim1, dim2, N, M, deltaX1, V, width, stride) \
	center_quadratic_dual_deltaX_kernel_4<LBY, LBX>\
		<<< dim3(MIN_int32(M>>LBX>>LTX, GRID_MAX),\
                 MIN_int32(N>>LBY>>LTY, GRID_MAX)),\
            dim3(1<<LBX, 1<<LBY), 0, stream >>>\
			(deltaY, X1, X2, k11, k12, k22, k1, k2, C, dim1, dim2, N, M, deltaX1, V, width, stride)

#endif


#ifndef CENTER_QUADRATIC_DUAL_DELTAX_KERNEL_4
#define CENTER_QUADRATIC_DUAL_DELTAX_KERNEL_4

//(1) deltaX1 = deltaY * ((2*k11)*X1 + k12*X2 + k1)
//(2) deltaX2 = deltaY * ((2*k22)*X2 + k12*X1 + k2)
template<int LBY, int LBX>
__global__ void center_quadratic_dual_deltaX_kernel_4(
	const float* __restrict__ deltaY,//[dim0, dim1, dim2]
	const float* __restrict__ X1,//[dim0, dim1, dim2]
	const float* __restrict__ X2,//[dim0,       dim2]
	float k11, float k12, float k22,
	float k1, float k2, float C,
	int dim1, int dim2, int N, int M,//N = d1, M = dim0 * dim2
	float* __restrict__ deltaX1,//[dim0, dim1, dim2]
	float* __restrict__ V,
	int width, int stride)
{
	const int by = blockIdx.y, bx = blockIdx.x;
	const int ty = threadIdx.y, tx = threadIdx.x;

	__shared__ float2 As[1 << LBX][(2 << LBY) + 2];

	//------[parallel element num = 4]--------------------------------------
	const int Y0 = (by << LBY) + ty, stepY = (gridDim.y << LBY);
	const int X0 = (bx << LBX) + tx, stepX = (gridDim.x << LBX);
	const int stepX4 = stepX << 2, M4 = (M >> 2) << 2;
	k22 *= 2.0f; k11 *= 2.0f;
	for (int x4 = X0 << 2; x4 < M4; x4 += stepX4)//x4 -> M[d0, d2]
	{
		//------compute area[thread reduce: 4 local result]-----------------
		float4 x2 = *(float4*)(X2 + x4);//[d0, d2]
		const int d0 = x4 / dim2, d2 = x4 - d0 * dim2;
		float4 c = F32_4_0, v = F32_4_0;//Kuhan summary
		for (int d1 = Y0; d1 < N; d1 += stepY)//y -> N[d1]
		{
			const int xoffset = (d0*dim1 + d1)*dim2 + d2;//[d0, d1, d2 + (0-3)]
			float4 x1 = *(float4*)(X1 + xoffset);//X1[y, x4]
			float4 dy = *(float4*)(deltaY + xoffset);//X2[y, x4]
			
			float4 dx1;//(1) deltaX1 = deltaY * ((2*k11)*X1 + k12*X2 + k1)
			dx1.x = dy.x * (k11 * x1.x + k12 * x2.x + k1);
			dx1.y = dy.y * (k11 * x1.y + k12 * x2.y + k1);
			dx1.z = dy.z * (k11 * x1.z + k12 * x2.z + k1);
			dx1.w = dy.w * (k11 * x1.w + k12 * x2.w + k1);

			float4 dx2;//(2) deltaX2 = deltaY * ((2*k22)*X2 + k12*X1 + k2)
			dx2.x = dy.x * (k22 * x2.x + k12 * x1.x + k2);
			dx2.y = dy.y * (k22 * x2.y + k12 * x1.y + k2);
			dx2.z = dy.z * (k22 * x2.z + k12 * x1.z + k2);
			dx2.w = dy.w * (k22 * x2.w + k12 * x1.w + k2);
			Kahan_simdAdd4(v, dx2, c);//v = v + dx2

			*(float4*)(deltaX1 + xoffset) = dx1;
		}
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
			*(float2*)(&get(V, by, xindex2, M)) = v1;//V[by: dim1 -> HV, x4: dim0*dim2]
		}
		__syncthreads();
	}
}

#endif


//======[integration]============================================
#ifndef CENTER_QUADRATIC_DUAL_DELTAX_STAGE
#define CENTER_QUADRATIC_DUAL_DELTAX_STAGE

//M % 4 == 0, M >= 4
void __center_quadratic_dual_deltaX_stage(cudaStream_t stream,
	const float* deltaY,//[dim0, dim1, dim2]
	const float* X1,//[dim0, dim1, dim2]
	const float* X2,//[dim0,       dim2]
	float k11, float k12, float k22,
	float k1, float k2, float C,
	int dim1, int dim2, int N, int M,
	float* deltaX1,//[dim0, dim1, dim2]
	float* V, int width, int stride)
{
	if (M > 15) {
		if (N > 63) { center_quadratic_dual_deltaX4(stream, 3, 2, 3, 2, deltaY, X1, X2, k11, k12, k22, k1, k2, C, dim1, dim2, N, M, deltaX1, V, width, stride); return; }//[64, 16]
		if (N > 31) { center_quadratic_dual_deltaX4(stream, 3, 2, 2, 2, deltaY, X1, X2, k11, k12, k22, k1, k2, C, dim1, dim2, N, M, deltaX1, V, width, stride); return; }//[32, 16]
		if (N > 15) { center_quadratic_dual_deltaX4(stream, 3, 2, 1, 2, deltaY, X1, X2, k11, k12, k22, k1, k2, C, dim1, dim2, N, M, deltaX1, V, width, stride); return; }//[16, 16]
	}
	if (M > 7) {
		if (N > 127) { center_quadratic_dual_deltaX4(stream, 4, 1, 3, 2, deltaY, X1, X2, k11, k12, k22, k1, k2, C, dim1, dim2, N, M, deltaX1, V, width, stride); return; }//[ 64, 8]
		if (N >  63) { center_quadratic_dual_deltaX4(stream, 4, 1, 2, 2, deltaY, X1, X2, k11, k12, k22, k1, k2, C, dim1, dim2, N, M, deltaX1, V, width, stride); return; }//[ 32, 8]
		if (N >  31) { center_quadratic_dual_deltaX4(stream, 4, 1, 1, 2, deltaY, X1, X2, k11, k12, k22, k1, k2, C, dim1, dim2, N, M, deltaX1, V, width, stride); return; }//[ 16, 8]
	}
	center_quadratic_dual_deltaX4_small(stream, deltaY, X1, X2, k11, k12, k22, k1, k2, C, dim1, dim2, N, M, deltaX1, V, width, stride);
}

#endif


#ifndef CENTER_QUADRATIC_DUAL_DELTAX
#define CENTER_QUADRATIC_DUAL_DELTAX

//N = height, M = width
//(1) height = dim1, height % 4 != 0
//(2) width = dim0*dim2, width % 4 = 0
int __center_quadratic_dual_deltaX(cudaStream_t stream,
	const float* deltaY,//[dim0, dim1, dim2]
	const float* X1,//[dim0, dim1, dim2]
	const float* X2,//[dim0,       dim2]
	float k11, float k12, float k22,
	float k1, float k2, float C,
	int dim0, int dim1, int dim2,
	float* deltaX1,//[dim0, dim1, dim2]
	float* V, 
	float* deltaX2,//[dim0,       dim2]
	int width, int stride,
	int partNum)
{
	int N = dim1, M = dim0 * dim2;
	int nextN = center_nextN(N, M);//N <= HV
	if (nextN <= partNum) {//V.address = NULL, only 1 stage, write result to Y
		__center_quadratic_dual_deltaX_stage(stream, deltaY, X1, X2, k11, k12, k22, k1, k2, C, dim1, dim2, N, M, deltaX1, deltaX2, width, stride);
		return nextN;
	}

	//V.address != NULL, at least 2 stages
	__center_quadratic_dual_deltaX_stage(stream, deltaY, X1, X2, k11, k12, k22, k1, k2, C, dim1, dim2, N, M, deltaX1, V, width, stride);

	N = nextN, nextN = field_nextN(N, M);
	while (nextN > partNum)//end: nextN <= partNum
	{
		__field_sum_stage(stream, V, N, M, V, width, stride);
		N = nextN, nextN = field_nextN(nextN, M);
	}
	__field_sum_stage(stream, V, N, M, deltaX2, width, stride);//the last stage
	return nextN;
}

#endif

#endif
