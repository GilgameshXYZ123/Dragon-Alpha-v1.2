#pragma once

#ifndef FIELD_BATCH_NORM_DELTA_AB_V2_H
#define FIELD_BATCH_NORM_DELTA_AB_V2_H

//(1) field_num % A.width ==0
//(2) field_num -> field_stride = field_num / width * stride
//(3) A.length  % feature_num   == 0
//(4) A.lengthv % fieled_stride == 0
//(5) row_num = A.lengthv / field_stride
//(6) [y, x] -> [row_num, field_stride]
//(7) [N, M] = [row_num, field_stride], A.lengthv = N*M
//(8) V2: holdX(), X is not changed
//(9) affine = true || false
//<1> (deltaXp1 = deltaB) = sum_each_field: deltaY
//<2> (deltaXp2 = deltaA) = sum_each_field: deltaY * Xnorm
#ifndef FIELD_BATCH_NORM_DELTA_AB_V2_CALL
#define FIELD_BATCH_NORM_DELTA_AB_V2_CALL

//[16, 4]
#define field_batchNorm_deltaAB_v2_k4_small(stream, deltaY, X, X_mean, X_var, eps, N, M, deltaA, deltaB, width, stride) \
	field_batchNorm_deltaAB_v2_kernel_4<3, 3>\
		<<< dim3(MIN_int32((M + 31)>>5, GRID_MAX),\
                 MIN_int32((N + 63)>>6, GRID_MAX)),\
            dim3(8, 8), 0, stream>>>\
			(deltaY, X, X_mean, X_var, eps, N, M, deltaA, deltaB, width, stride)

//LTX >=2
#define field_batchNorm_deltaAB_v2_k4(stream, LBY, LBX, LTY, LTX, deltaY, X, X_mean, X_var, eps, N, M, deltaA, deltaB, width, stride) \
	field_batchNorm_deltaAB_v2_kernel_4<LBY, LBX>\
		<<< dim3(MIN_int32(M>>LBX>>LTX, GRID_MAX),\
                 MIN_int32(N>>LBY>>LTY, GRID_MAX)),\
            dim3(1<<LBX, 1<<LBY), 0, stream >>>\
			(deltaY, X, X_mean, X_var, eps, N, M, deltaA, deltaB, width, stride)

#endif


#ifndef FIELD_BATCH_NORM_DELTA_AB_V2_KERNEL_4
#define FIELD_BATCH_NORM_DELTA_AB_V2_KERNEL_4

//======[Document]============================================
//<1> N = batch_size =  field_length = lengthv / row_lengthv
//<2> M = row_lengthv
//<3> dims: Y[N, M], X[N, M], X_mean[M], X_var[M], A[M], B[M]
//<4> dims: deltaY[N, M], deltaX[N, M], deltaXp1[M], deltaXp2[M]
//
//[Backward Propagation]
//STEP:
//<1> X_rstd = rsqrt(X_var + eps)
//<2> X_norm = (X - X_mean) * X_rstd
//<3> (deltaXp2 = deltaA) = field_sum: deltaY * X_norm
//<4> (deltaXp1 = deltaB) = field_sum: deltaY
//======[Document]============================================

template<int LBY, int LBX>
__global__ void field_batchNorm_deltaAB_v2_kernel_4(
	const float* __restrict__ deltaY,
	const float* __restrict__ X,
	const float* __restrict__ X_mean,
	const float* __restrict__ X_var, float eps,
	int N, int M,
	float* __restrict__ deltaA,//deltaA = deltaXp2
	float* __restrict__ deltaB,//deltaB = deltaXp1
	int width, int stride)
{
	const int by = blockIdx.y, bx = blockIdx.x;
	const int ty = threadIdx.y, tx = threadIdx.x;

	__shared__ float4 As[1 << LBX][(2 << LBY) + 1];

	//------[parallel field num = 4]-----------------------------------------
	const int offsetY = (by << LBY) + ty, stepY = (gridDim.y << LBY);
	const int offsetX = (bx << LBX) + tx, stepX = (gridDim.x << LBX);
	const int stepX4 = stepX << 2, M4 = (M >> 2) << 2;

	float2 table[2]; table[0] = F32_2_0;//(x_var == 0) may cause NaN
	for (int x4 = offsetX << 2; x4 < M4; x4 += stepX4)
	{
		//------compute area[thread reduce: 4 local result]------------------
		float4 v0 = F32_4_0, c0 = F32_4_0;//v0 = field_sum: deltaY
		float4 v1 = F32_4_0, c1 = F32_4_0;//v1 = field_sum: deltaY * X
		for (int y = offsetY; y < N; y += stepY)
		{
			const int src_offset = y * M + x4;
			const float4 dy = *(float4*)(deltaY + src_offset);//deltaY[y, x4]
			const float4 xv = *(float4*)(X + src_offset);//deltaX[y, x4]

			float4 dv1;//(deltaXp2 = deltaA) = field_sum: deltaY * X_norm
			dv1.x = dy.x * xv.x;
			dv1.y = dy.y * xv.y;
			dv1.z = dy.z * xv.z;
			dv1.w = dy.w * xv.w;

			Kahan_simdAdd4(v0,  dy, c0);//(deltaXp1 = deltaB) = field_sum: deltaY
			Kahan_simdAdd4(v1, dv1, c1);//field sum for: (deltaXp2 = deltaA)
		}
		As[tx][(ty << 1)    ] = float4{ v0.x, v0.y, v1.x, v1.y };//[deltaXp1, deltaB]
		As[tx][(ty << 1) + 1] = float4{ v0.z, v0.w, v1.z, v1.w };//[deltaXp2, deltaA]
		__syncthreads();

		//------compute area[block reduce: 4 global result]------------------
		int yIdx;
		float4 scs[2]{//overlap the error in same field
			float4{ c0.x, c0.y, c1.x, c1.y },//yIdx % 2 == 0
			float4{ c0.z, c0.w, c1.z, c1.w } //yIdx % 2 != 0
		};

		if (LBY >= 6) {//BLOCK_Y = 64
			yIdx = ((ty & 31) << 1) + (ty >> 5);
			if (ty < 64) {//128 -> 64
				float4 sv1 = As[tx][yIdx], sv2 = As[tx][yIdx + 64], sc = scs[yIdx & 1];
				Kahan_simdAdd4(sv1, sv2, sc); As[tx][yIdx] = sv1;
			}
			__syncthreads();
		}

		if (LBY >= 5) {//BLOCK_Y = 32
			yIdx = ((ty & 15) << 1) + (ty >> 4);
			if (ty < 32) {//64 -> 32
				float4 sv1 = As[tx][yIdx], sv2 = As[tx][yIdx + 32], sc = scs[yIdx & 1];
				Kahan_simdAdd4(sv1, sv2, sc); As[tx][yIdx] = sv1;
			}
			__syncthreads();
		}

		if (LBY >= 4) {//BLOCK_Y = 16
			yIdx = ((ty & 7) << 1) + (ty >> 3);
			if (ty < 16) {//32 -> 16
				float4 sv1 = As[tx][yIdx], sv2 = As[tx][yIdx + 16], sc = scs[yIdx & 1];
				Kahan_simdAdd4(sv1, sv2, sc); As[tx][yIdx] = sv1;
			}
			__syncthreads();
		}

		yIdx = ((ty & 3) << 1) + (ty >> 2);//in all cases: LBY >= 3
		if (ty < 8) {//16 -> 8
			float4 sv1 = As[tx][yIdx], sv2 = As[tx][yIdx + 8], sc = scs[yIdx & 1];
			Kahan_simdAdd4(sv1, sv2, sc); As[tx][yIdx] = sv1;
		}
		__syncthreads();

		yIdx = ((ty & 1) << 1) + (ty >> 1);
		if (ty < 4) {//8 -> 4
			float4 sv1 = As[tx][yIdx], sv2 = As[tx][yIdx + 4], sc = scs[yIdx & 1];
			Kahan_simdAdd4(sv1, sv2, sc); As[tx][yIdx] = sv1;
		}
		__syncthreads();

		yIdx = ((ty & 0) << 1) + (ty >> 0);
		if (ty < 2) {
			float4 sv1 = As[tx][yIdx], sv2 = As[tx][yIdx + 2], sc = scs[yIdx & 1];
			Kahan_simdAdd4(sv1, sv2, sc);//sv1 += sv2

			float2 dB = float2{ sv1.x, sv1.y };//deltaXp1 = deltaB
			float2 dA = float2{ sv1.z, sv1.w };//deltaXp2 = deltaA

			//(deltaXp2 = deltaA) = field_sum: deltaY * (X - X_mean) * X_rstd
			const int xindex2 = x4 + (ty << 1);//with the same tx & field
			const float2 x_mean = *(float2*)(X_mean + xindex2);
			const float2 x_var  = *(float2*)(X_var  + xindex2);
			dA.x = (dA.x - dB.x * x_mean.x) * rsqrtf(x_var.x + eps);
			dA.y = (dA.y - dB.y * x_mean.y) * rsqrtf(x_var.y + eps);
			within_width_zero_nan2(dB, xindex2, table, stride, width);
			within_width_zero_nan2(dA, xindex2, table, stride, width);

			const int dst_index = by * M + xindex2;
			*(float2*)(deltaB + dst_index) = dB;//(deltaXp1 = deltaB)[by, xindex2]
			*(float2*)(deltaA + dst_index) = dA;//(deltaXp2 = deltaA)[by, xindex2]
		}
		__syncthreads();
	}
}

#endif


//======[integration]========================================
#ifndef FIELD_BATCH_NORM_DELTA_AB_V2_STAGE
#define FIELD_BATCH_NORM_DELTA_AB_V2_STAGE

void __field_batchNorm_deltaAB_v2_stage(cudaStream_t stream,
	const float* deltaY,
	const float* X,//V2: holdX(), X is not changed
	const float* X_mean,
	const float* X_var, float eps,
	int N, int M,
	float* deltaA,//deltaA = deltaXp2
	float* deltaB,//deltaB = deltaXp1
	int width, int stride)
{
	if (M > 15) {
		if (N > 63) { field_batchNorm_deltaAB_v2_k4(stream, 3, 2, 3, 2, deltaY, X, X_mean, X_var, eps, N, M, deltaA, deltaB, width, stride); return; }//[64, 16]
		if (N > 31) { field_batchNorm_deltaAB_v2_k4(stream, 3, 2, 2, 2, deltaY, X, X_mean, X_var, eps, N, M, deltaA, deltaB, width, stride); return; }//[32, 16]
		if (N > 15) { field_batchNorm_deltaAB_v2_k4(stream, 3, 2, 1, 2, deltaY, X, X_mean, X_var, eps, N, M, deltaA, deltaB, width, stride); return; }//[16, 16]
	}
	if (M > 7) {
		if (N > 127) { field_batchNorm_deltaAB_v2_k4(stream, 4, 1, 3, 2, deltaY, X, X_mean, X_var, eps, N, M, deltaA, deltaB, width, stride); return; }//[ 64, 8]
		if (N > 63) { field_batchNorm_deltaAB_v2_k4(stream, 4, 1, 2, 2, deltaY, X, X_mean, X_var, eps, N, M, deltaA, deltaB, width, stride); return; }//[ 32, 8]
		if (N > 31) { field_batchNorm_deltaAB_v2_k4(stream, 4, 1, 1, 2, deltaY, X, X_mean, X_var, eps, N, M, deltaA, deltaB, width, stride); return; }//[ 16, 8]
	}
	field_batchNorm_deltaAB_v2_k4_small(stream, deltaY, X, X_mean, X_var, eps, N, M, deltaA, deltaB, width, stride);
}

#endif 


#ifndef FIELD_BATCH_NORM_DELTA_AB_V2
#define FIELD_BATCH_NORM_DELTA_AB_V2

int __field_batchNorm_deltaAB_v2(JNIEnv *env,
	cudaStream_t stream1, cudaStream_t stream2,
	const float* deltaY,
	const float* X,//V2: holdX(), X is not changed
	const float* X_mean,
	const float* X_var, float eps,
	int N, int M,
	float* deltaA_buf, float* deltaA,//deltaA = deltaXp2
	float* deltaB_buf, float* deltaB,//deltaB = deltaXp1
	int width, int stride,
	int partNum)
{
	int nextN = field_nextN(N, M);//N <= HV
	if (nextN <= partNum) {//V.address = NULL, only 1 stage, write result to Y
		__field_batchNorm_deltaAB_v2_stage(stream1,
			deltaY, X, X_mean, X_var, eps, N, M,
			deltaA,//deltaXp2 = deltaA
			deltaB,//deltaXp1 = deltaB
			width, stride);
		return nextN;
	}

	//V.address != NULL, at least 2 stages
	__field_batchNorm_deltaAB_v2_stage(stream1,
		deltaY, X, X_mean, X_var, eps, N, M,
		deltaA_buf,
		deltaB_buf,
		width, stride);

	//====stream2 shold wait stream1======================================================
	cudaEvent_t event; cudaError_t error;
	error = cudaEventCreate(&event, cudaEventDisableTiming); handleError(error);
	error = cudaEventRecord(event, stream1); handleError(error);
	error = cudaStreamWaitEvent(stream2, event, cudaEventWaitDefault); handleError(error);
	//====stream2 shold wait stream1======================================================

	N = nextN, nextN = field_nextN(N, M);//N % 4 == 0, N >= 4
	while (nextN > partNum) {//end: nextN <= partNum
		__field_sum_stage(stream1, deltaA_buf, N, M, deltaA_buf, width, stride);
		__field_sum_stage(stream2, deltaB_buf, N, M, deltaB_buf, width, stride);
		N = nextN, nextN = field_nextN(nextN, M);
	}
	__field_sum_stage(stream1, deltaA_buf, N, M, deltaA, width, stride);//the last stage
	__field_sum_stage(stream2, deltaB_buf, N, M, deltaB, width, stride);//the last stage

	error = cudaEventDestroy(event); handleError(error);
	return nextN;
}

#endif

#endif