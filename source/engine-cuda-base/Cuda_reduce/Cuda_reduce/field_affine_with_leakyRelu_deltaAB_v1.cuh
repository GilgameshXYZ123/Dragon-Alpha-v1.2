#pragma once

#ifndef FIELD_AFFINE_WITH_LEAKY_RELU_DELTA_AB_V1_H
#define FIELD_AFFINE_WITH_LEAKY_RELU_DELTA_AB_V1_H

//(1) field_num % A.width ==0
//(2) field_num -> field_stride = field_num / width * stride
//(3) A.length % feature_num == 0
//(4) A.lengthv % fieled_stride == 0
//(5) row_num = A.lengthv / field_stride
//(6) [y, x] -> [row_num, field_stride]
//(7) [N, M] = [row_num, field_stride], A.lengthv = N*M
//(8) V1: holdY(), Y is not changed
//(9) Can be used for layerNorm, batchNrom, sqBatchNorm (affine = true)
#ifndef FIELD_AFFINE_WITH_LEAKY_RELU_DELTA_AB_V1_CALL
#define FIELD_AFFINE_WITH_LEAKY_RELU_DELTA_AB_V1_CALL

//[16, 4]
#define field_affine_with_leakyRelu_deltaAB_v1_k4_small(stream, deltaY, Y, k, A, B, N, M, deltaA, deltaB) \
	field_affine_with_leakyRelu_deltaAB_v1_kernel_4<3, 3>\
		<<< dim3(MIN_int32((M + 31)>>5, GRID_MAX),\
                 MIN_int32((N + 63)>>6, GRID_MAX)),\
            dim3(8, 8), 0, stream>>>\
			(deltaY, Y, k, A, B, N, M, deltaA, deltaB, width, stride)

//LTX >=2
#define field_affine_with_leakyRelu_deltaAB_v1_k4(stream, LBY, LBX, LTY, LTX, deltaY, Y, k, A, B, N, M, deltaA, deltaB) \
	field_affine_with_leakyRelu_deltaAB_v1_kernel_4<LBY, LBX>\
		<<< dim3(MIN_int32(M>>LBX>>LTX, GRID_MAX),\
                 MIN_int32(N>>LBY>>LTY, GRID_MAX)),\
            dim3(1<<LBX, 1<<LBY), 0, stream >>>\
			(deltaY, Y, k, A, B, N, M, deltaA, deltaB, width, stride)

#endif


#ifndef FIELD_AFFINE_WITH_LEAKY_RELU_DELTA_AB_V1_KERNEL_4
#define FIELD_AFFINE_WITH_LEAKY_RELU_DELTA_AB_V1_KERNEL_4

//======[Document]============================================
//<1> N = batch_size =  field_length = lengthv / row_lengthv
//<2> M = row_lengthv
//<3> dims: Y[N, M], X[N, M], X_mean[M], X_mean[M], A[M], B[M]
//<4> dims: deltaY[N, M], deltaX[N, M], deltaXp1[M], deltaXp2[M]
//
//[Forward Propagation]
//<1> affine: 
//	(1) Y1 = A*X + B
//  (2) Y2 = leaky_relu(Y1) = Y
//<2> Norm: 
//	(1) Y1 = A*X_norm + B
//  (2) Y2 = leaky_relu(Y1) = Y. obviously: sign(Y1) = sign(Y2)
//
//[Backward Propagation]
//STEP:
//(1) rK = 1 / k
//(2) Y1 = Y2 * (Y2 > 0 ? 1 : rk)
//(3) deltaY1 = deltaY2 * (Y2 > 0 ? 1 : k), deltaY2 = deltaY
//(4) (X_norm || X) = (Y1 - B) / A
//(5) (deltaXp2 = deltaA) = field_sum: deltaY1 * X_norm
//(6) (deltaXp1 = deltaB) = field_sum: deltaY1
//======[Document]============================================

template<int LBY, int LBX>
__global__ void field_affine_with_leakyRelu_deltaAB_v1_kernel_4(
	const float* __restrict__ deltaY,
	const float* __restrict__ Y, float k,
	const float* __restrict__ A,
	const float* __restrict__ B, int N, int M,
	      float* __restrict__ deltaA,//deltaA = deltaXp2
	      float* __restrict__ deltaB,//deltaB = deltaXp1
	int width, int stride)
{
	const int by = blockIdx.y, bx = blockIdx.x;
	const int ty = threadIdx.y, tx = threadIdx.x;

	__shared__ float4 As[1 << LBX][(2 << LBY) + 1];

	//------[parallel field num = 4]----------------------------------------
	const int offsetY = (by << LBY) + ty, stepY = (gridDim.y << LBY);
	const int offsetX = (bx << LBX) + tx, stepX = (gridDim.x << LBX);
	const int stepX4 = stepX << 2, M4 = (M >> 2) << 2;

	const float rK = 1.0f / k;
	float2 table[2]; table[0] = F32_2_0;//(A == 0) may cause NaN
	for (int x4 = offsetX << 2; x4 < M4; x4 += stepX4)
	{
		//------compute area[thread reduce: 4 local result]-----------------
		float4 v0 = F32_4_0, c0 = F32_4_0;//v0 = field_sum: deltaY
		float4 v1 = F32_4_0, c1 = F32_4_0;//v1 = field_sum: deltaY * Y
		for (int y = offsetY; y < N; y += stepY)
		{
			const int src_offset = y * M + x4;
			const float4 dy2 = *(float4*)(deltaY + src_offset);//deltaY[y, x4]
			const float4 y2v = *(float4*)(Y + src_offset);//Y[y, x4]

			char4 flag;//flag = (Y2 > 0)
			flag.x = (y2v.x > 0.0f);
			flag.y = (y2v.y > 0.0f);
			flag.z = (y2v.z > 0.0f);
			flag.w = (y2v.w > 0.0f);

			float4 y1v;//Y1 = Y2 * (Y2 > 0 ? 1 : rK)
			y1v.x = y2v.x * (flag.x + !flag.x*rK);
			y1v.y = y2v.y * (flag.y + !flag.y*rK);
			y1v.z = y2v.z * (flag.z + !flag.z*rK);
			y1v.w = y2v.w * (flag.w + !flag.w*rK);

			float4 dy1;//deltaY1 = deltaY2 * (Y2 > 0 ? 1 : k)
			dy1.x = dy2.x * (flag.x + !flag.x*k);
			dy1.y = dy2.y * (flag.y + !flag.y*k);
			dy1.z = dy2.z * (flag.z + !flag.z*k);
			dy1.w = dy2.w * (flag.w + !flag.w*k);
			
			float4 dv1;//(deltaXp2 = deltaA) = field_sum: deltaY1 * X_norm(Y1)
			dv1.x = dy1.x * y1v.x;
			dv1.y = dy1.y * y1v.y;
			dv1.z = dy1.z * y1v.z;
			dv1.w = dy1.w * y1v.w;

			Kahan_simdAdd4(v0, dy1, c0);//(deltaXp1 = deltaB) = field_sum: deltaY1
			Kahan_simdAdd4(v1, dv1, c1);//field sum for: (deltaXp2 = deltaA)
		}
		As[tx][(ty << 1)    ] = float4{ v0.x, v0.y, v1.x, v1.y };//[deltaXp1, deltaB]
		As[tx][(ty << 1) + 1] = float4{ v0.z, v0.w, v1.z, v1.w };//[deltaXp2, deltaA]
		__syncthreads();

		//compute area[block reduce: 4 global result]------------------
		int yIdx;
		float4 scs[2]{//with same tx: overlap the error in same field
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
		if (ty < 2) {//4 -> 2, save
			float4 sv1 = As[tx][yIdx], sv2 = As[tx][yIdx + 2], sc = scs[(yIdx & 1)];
			Kahan_simdAdd4(sv1, sv2, sc);//v1 += v2

			float2 dB = float2{ sv1.x, sv1.y };//deltaXp1 = deltaB
			float2 dA = float2{ sv1.z, sv1.w };//deltaXp2 = deltaA

			//(deltaXp2 = deltaA) = field_sum: deltaY1 * (Y1 - B) / A
			const int xindex2 = x4 + (ty << 1);
			const float2 a = *(float2*)(A + xindex2);
			const float2 b = *(float2*)(B + xindex2);
			dA.x = (dA.x - b.x * dB.x) / a.x;
			dA.y = (dA.y - b.y * dB.y) / a.y;
			within_width_zero_nan2(dB, xindex2, table, stride, width);
			within_width_zero_nan2(dA, xindex2, table, stride, width);

			const int dst_index = by * M + xindex2;
			*(float2*)(deltaB + dst_index) = dB;//deltaXp1, deltaB[by, xindex2]
			*(float2*)(deltaA + dst_index) = dA;//deltaXp2, deltaA[by, xindex2]
		}
		__syncthreads();
	}
}

#endif


//======[integration]========================================
#ifndef FIELD_AFFINE_WITH_LEAKY_RELU_DELTA_AB_V1_STAGE
#define FIELD_AFFINE_WITH_LEAKY_RELU_DELTA_AB_V1_STAGE

//M % 4 == 0, M >= 4
void __field_affine_with_leakyRelu_deltaAB_v1_stage(cudaStream_t stream,
	const float* deltaY,
	const float* Y, float k,//V1: holdY(), Y is not changed
	const float* A, 
	const float* B, int N, int M,
	      float* deltaA,//deltaA = deltaXp2
	      float* deltaB,//deltaB = deltaXp1
	int width, int stride)
{
	if (M > 15) {
		if (N > 63) { field_affine_with_leakyRelu_deltaAB_v1_k4(stream, 3, 2, 3, 2, deltaY, Y, k, A, B, N, M, deltaA, deltaB); return; }//[64, 16]
		if (N > 31) { field_affine_with_leakyRelu_deltaAB_v1_k4(stream, 3, 2, 2, 2, deltaY, Y, k, A, B, N, M, deltaA, deltaB); return; }//[32, 16]
		if (N > 15) { field_affine_with_leakyRelu_deltaAB_v1_k4(stream, 3, 2, 1, 2, deltaY, Y, k, A, B, N, M, deltaA, deltaB); return; }//[16, 16]
	}
	if (M > 7) {
		if (N > 127) { field_affine_with_leakyRelu_deltaAB_v1_k4(stream, 4, 1, 3, 2, deltaY, Y, k, A, B, N, M, deltaA, deltaB); return; }//[ 64, 8]
		if (N >  63) { field_affine_with_leakyRelu_deltaAB_v1_k4(stream, 4, 1, 2, 2, deltaY, Y, k, A, B, N, M, deltaA, deltaB); return; }//[ 32, 8]
		if (N >  31) { field_affine_with_leakyRelu_deltaAB_v1_k4(stream, 4, 1, 1, 2, deltaY, Y, k, A, B, N, M, deltaA, deltaB); return; }//[ 16, 8]
	}
	field_affine_with_leakyRelu_deltaAB_v1_k4_small(stream, deltaY, Y, k, A, B, N, M, deltaA, deltaB);
}

#endif 


#ifndef FIELD_AFFINE_WITH_LEAKY_RELU_DELTA_AB_V1
#define FIELD_AFFINE_WITH_LEAKY_RELU_DELTA_AB_V1

int __field_affine_with_leakyRelu_deltaAB_v1(JNIEnv *env,
	cudaStream_t stream1, cudaStream_t stream2,
	const float* deltaY,
	const float* Y, float k,//V1: holdY(), Y is not changed
	const float* A, 
	const float* B, int N, int M,
	      float* deltaA_buf, float* deltaA,//deltaA = deltaXp2
	      float* deltaB_buf, float* deltaB,//deltaB = deltaXp1
	int width, int stride,
	int partNum)
{
	int nextN = field_nextN(N, M);//N <= HV
	if (nextN <= partNum) {//V.address = NULL, only 1 stage, write result to Y
		__field_affine_with_leakyRelu_deltaAB_v1_stage(stream1,
			deltaY, Y, k, A, B, N, M,
			deltaA,//deltaA = deltaXp2
			deltaB,//deltaB = deltaXp1
			width, stride);
		return nextN;
	}

	//V.address != NULL, at least 2 stages
	__field_affine_with_leakyRelu_deltaAB_v1_stage(stream1,
		deltaY, Y, k, A, B, N, M,
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