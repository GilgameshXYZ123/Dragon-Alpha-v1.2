#pragma once

#ifndef FIELD_AFFINE_DELTA_A_V1_H
#define FIELD_AFFINE_DELTA_A_V1_H

//(1) field_num % A.width ==0
//(2) field_num -> field_stride = field_num / width * stride
//(3) A.length % feature_num == 0
//(4) A.lengthv % fieled_stride == 0
//(5) row_num = A.lengthv / field_stride
//(6) [y, x] -> [row_num, field_stride]
//(7) [N, M] = [row_num, field_stride], A.lengthv = N*M
//(8) V1: holdY(), Y is not changed
//(9) affine = true:
//Can be used for layerNorm, batchNrom, sqBatchNorm
#ifndef FIELD_AFFINE_DELTA_A_V1_CALL
#define FIELD_AFFINE_DELTA_A_V1_CALL

//[16, 4]
#define field_affine_deltaA_v1_k4_small(stream, deltaY, Y, A, B, N, M, deltaA, width, stride) \
	field_affine_deltaA_v1_kernel_4<3, 3>\
		<<< dim3(MIN_int32((M + 31)>>5, GRID_MAX),\
                 MIN_int32((N + 63)>>6, GRID_MAX)),\
            dim3(8, 8), 0, stream>>>\
			(deltaY, Y, A, B, N, M, deltaA, width, stride)
//LTX >=2
#define field_affine_deltaA_v1_k4(stream, LBY, LBX, LTY, LTX, deltaY, Y, A, B, N, M, deltaA, width, stride) \
	field_affine_deltaA_v1_kernel_4<LBY, LBX>\
		<<< dim3(MIN_int32(M>>LBX>>LTX, GRID_MAX),\
                 MIN_int32(N>>LBY>>LTY, GRID_MAX)),\
            dim3(1<<LBX, 1<<LBY), 0, stream >>>\
			(deltaY, Y, A, B, N, M, deltaA, width, stride)

#endif


#ifndef FIELD_AFFINE_DELTA_A_V1_KERNEL_4
#define FIELD_AFFINE_DELTA_A_V1_KERNEL_4

//======[Document]============================================
//<1> N = batch_size =  field_length = lengthv / row_lengthv
//<2> M = row_lengthv
//<3> dims: Y[N, M], X[N, M], X_mean[M], X_var[M], A[M], B[M]
//<4> dims: deltaY[N, M], deltaX[N, M], deltaXp1[M], deltaXp2[M]
//
//[Forward Propagation]
//<1> affine: Y = A*X + B
//<2> Norm: Y = A*X_norm + B
//
//[Backward Propagation]
//STEP:
//<1> (X_norm || X) = (Y - B) / A
//<2> (deltaXp2 = deltaA) = field_sum: deltaY * X_norm
//======[Document]============================================

template<int LBY, int LBX>
__global__ void field_affine_deltaA_v1_kernel_4(
	const float* __restrict__ deltaY,
	const float* __restrict__ Y,
	const float* __restrict__ A,
	const float* __restrict__ B,
	int N, int M,
	float* __restrict__ deltaA,//deltaA = deltaXp2
	int width, int stride)
{
	const int by = blockIdx.y, bx = blockIdx.x;
	const int ty = threadIdx.y, tx = threadIdx.x;

	__shared__ float2 As[1 << LBX][(2 << LBY) + 2];

	//------[parallel field num = 4]----------------------------------------
	const int offsetY = (by << LBY) + ty, stepY = (gridDim.y << LBY);
	const int offsetX = (bx << LBX) + tx, stepX = (gridDim.x << LBX);
	const int stepX4 = stepX << 2, M4 = (M >> 2) << 2;

	float2 table[2]; table[0] = F32_2_0;//(A == 0) may cause NaN
	for (int x4 = offsetX << 2; x4 < M4; x4 += stepX4)
	{
		const float4 b = *(float4*)(B + x4);

		//------compute area[thread reduce: 4 local result]-----------------
		float4 c0 = F32_4_0;//solve the errors
		float4 v0 = F32_4_0;//thread reduce: 4 local result (deltaXp2, deltaA)
		for (int y = offsetY; y < N; y += stepY)
		{
			const int src_offset = y * M + x4;
			const float4 dy = *(float4*)(deltaY + src_offset);//deltaY[y, x4]
			const float4 yv = *(float4*)(Y + src_offset);//Y[y, x4]

			float4 dv0;//(deltaXp2 = deltaA) = field_sum: deltaY * X_norm
			dv0.x = dy.x * (yv.x - b.x);
			dv0.y = dy.y * (yv.y - b.y);
			dv0.z = dy.z * (yv.z - b.z);
			dv0.w = dy.w * (yv.w - b.w);
			
			Kahan_simdAdd4(v0, dv0, c0);//field sum for: (deltaXp2 = deltaA)
		}
		*(float4*)(&As[tx][ty << 1]) = v0;//[deltaXp1, deltaB]
		__syncthreads();

		//------compute area[block reduce: 4 global result]-----------------
		int yIdx;
		float2 scs[2]{//with same tx: overlap the error in same field 
			float2{ c0.x, c0.y },//yIdx % 2 == 0
			float2{ c0.z, c0.w } //yIdx % 2 != 0
		};

		if (LBY >= 6) {//BLOCK_Y = 64
			yIdx = ((ty & 31) << 1) + (ty >> 5);
			if (ty < 64) {//128 -> 64
				float2 sv1 = As[tx][yIdx], sv2 = As[tx][yIdx + 64], sc = scs[yIdx & 1];
				Kahan_simdAdd2(sv1, sv2, sc); As[tx][yIdx] = sv1;
			}
			__syncthreads();
		}

		if (LBY >= 5) {//BLOCK_Y = 32
			yIdx = ((ty & 15) << 1) + (ty >> 4);
			if (ty < 32) {//64 -> 32
				float2 sv1 = As[tx][yIdx], sv2 = As[tx][yIdx + 32], sc = scs[yIdx & 1];
				Kahan_simdAdd2(sv1, sv2, sc); As[tx][yIdx] = sv1;
			}
			__syncthreads();
		}

		if (LBY >= 4) {//BLOCK_Y = 16
			yIdx = ((ty & 7) << 1) + (ty >> 3);
			if (ty < 16) {//32 -> 16
				float2 sv1 = As[tx][yIdx], sv2 = As[tx][yIdx + 16], sc = scs[yIdx & 1];
				Kahan_simdAdd2(sv1, sv2, sc); As[tx][yIdx] = sv1;
			}
			__syncthreads();
		}

		yIdx = ((ty & 3) << 1) + (ty >> 2);//in all cases: LBY >= 3
		if (ty < 8) {//16 -> 8
			float2 sv1 = As[tx][yIdx], sv2 = As[tx][yIdx + 8], sc = scs[yIdx & 1];
			Kahan_simdAdd2(sv1, sv2, sc); As[tx][yIdx] = sv1;
		}
		__syncthreads();

		yIdx = ((ty & 1) << 1) + (ty >> 1);
		if (ty < 4) {//8 -> 4
			float2 sv1 = As[tx][yIdx], sv2 = As[tx][yIdx + 4], sc = scs[yIdx & 1];
			Kahan_simdAdd2(sv1, sv2, sc); As[tx][yIdx] = sv1;
		}
		__syncthreads();

		yIdx = ((ty & 0) << 1) + (ty >> 0);
		if (ty < 2) {//4 -> 2, save
			float2 sv1 = As[tx][yIdx], sv2 = As[tx][yIdx + 2], sc = scs[(yIdx & 1)];
			Kahan_simdAdd2(sv1, sv2, sc);//sv1 += sv2

			//(deltaXp2 = deltaA) = field_sum: deltaY * (Y - B) / A
			const int xindex2 = x4 + (ty << 1);
			const float2 a = *(float2*)(A + xindex2);
			sv1.x = sv1.x / a.x;
			sv1.y = sv1.y / a.y;
			within_width_zero_nan2(sv1, xindex2, table, stride, width);

			const int dst_index = by * M + xindex2;//[by, xindex2]
			*(float2*)(deltaA + dst_index) = sv1;
		}
		__syncthreads();
	}
}

#endif


//======[integration]========================================
#ifndef FIELD_AFFINE_DELTA_A_V1_STAGE
#define FIELD_AFFINE_DELTA_A_V1_STAGE

//M % 4 == 0, M >= 4
void __field_affine_deltaA_v1_stage(cudaStream_t stream,
	const float* deltaY,
	const float* Y,//V1: holdY(), Y is nnot changed
	const float* A, const float* B,
	int N, int M, 
	float*  deltaA,//deltaA = deltaXp2
	int width, int stride)
{
	if (M > 15) {
		if (N > 63) { field_affine_deltaA_v1_k4(stream, 3, 2, 3, 2, deltaY, Y, A, B, N, M, deltaA, width, stride); return; }//[64, 16]
		if (N > 31) { field_affine_deltaA_v1_k4(stream, 3, 2, 2, 2, deltaY, Y, A, B, N, M, deltaA, width, stride); return; }//[32, 16]
		if (N > 15) { field_affine_deltaA_v1_k4(stream, 3, 2, 1, 2, deltaY, Y, A, B, N, M, deltaA, width, stride); return; }//[16, 16]
	}
	if (M > 7) {
		if (N > 127) { field_affine_deltaA_v1_k4(stream, 4, 1, 3, 2, deltaY, Y, A, B, N, M, deltaA, width, stride); return; }//[ 64, 8]
		if (N >  63) { field_affine_deltaA_v1_k4(stream, 4, 1, 2, 2, deltaY, Y, A, B, N, M, deltaA, width, stride); return; }//[ 32, 8]
		if (N >  31) { field_affine_deltaA_v1_k4(stream, 4, 1, 1, 2, deltaY, Y, A, B, N, M, deltaA, width, stride); return; }//[ 16, 8]
	}
	field_affine_deltaA_v1_k4_small(stream, deltaY, Y, A, B, N, M, deltaA, width, stride);
}

#endif 


#ifndef FIELD_AFFINE_DELTA_A_V1
#define FIELD_AFFINE_DELTA_A_V1

int __field_affine_deltaA_v1(cudaStream_t stream,
	const float* deltaY, 
	const float* Y,//V1: holdY(), Y is not changed
	const float* A, const float* B,
	int N, int M, 
	float* deltaA_buf, float* deltaA,//deltaA = deltaXp2
	int width, int stride,
	int partNum)
{
	int nextN = field_nextN(N, M);//N <= HV
	if (nextN <= partNum) {//V.address = NULL, only 1 stage, write result to Y
		__field_affine_deltaA_v1_stage(stream, 
			deltaY, Y, A, B, N, M, 
			deltaA,//deltaA = deltaXp2
			width, stride);
		return nextN;
	}
	
	//V.address != NULL, at least 2 stages
	__field_affine_deltaA_v1_stage(stream, deltaY, Y, A, B, N, M, 
		deltaA_buf, 
		width, stride);

	N = nextN, nextN = field_nextN(N, M);//N % 4 == 0, N >= 4
	while (nextN > partNum) {//end: nextN <= partNum
		__field_sum_stage(stream, deltaA_buf, N, M, deltaA_buf, width, stride);
		N = nextN, nextN = field_nextN(nextN, M);
	}
	__field_sum_stage(stream, deltaA_buf, N, M, deltaA, width, stride);//the last stage
	return nextN;
}

#endif

#endif