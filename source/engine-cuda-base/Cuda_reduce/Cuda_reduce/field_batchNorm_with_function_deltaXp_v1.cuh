#pragma once

#ifndef FIELD_BATCH_NORM_WITH_FUNCTION_DELTA_XP_V1_H
#define FIELD_BATCH_NORM_WITH_FUNCTION_DELTA_XP_V1_H

//(1) field_num % A.width ==0
//(2) field_num -> field_stride = field_num / width * stride
//(3) A.length  % feature_num   == 0
//(4) A.lengthv % fieled_stride == 0
//(5) row_num = A.lengthv / field_stride
//(6) [y, x] -> [row_num, field_stride]
//(7) [N, M] = [row_num, field_stride], A.lengthv = N*M
//(8) V2: holdX(), X is not changed
//(9) affine = false
//<1> deltaXp1 = sum_each_field: deltaY1
//<2> deltaXp2 = sum_each_field: deltaY1 * Xnorm
#ifndef FIELD_BATCH_NORM_WITH_FUNCTION_DELTA_XP_V1_CALL
#define FIELD_BATCH_NORM_WITH_FUNCTION_DELTA_XP_V1_CALL

//[16, 4]
#define field_batchNorm_with_function_deltaXp_v1_k4_small(stream, deltaY, Y, N, M, deltaXp1, deltaXp2) \
	field_batchNorm_with_function_deltaXp_v1_kernel_4<3, 3, fp32_func_type>\
		<<< dim3(MIN_int32((M + 31)>>5, GRID_MAX),\
                 MIN_int32((N + 63)>>6, GRID_MAX)),\
            dim3(8, 8), 0, stream>>>\
			(deltaY, Y, N, M, deltaXp1, deltaXp2, width, stride,\
				fp32_func_param0, fp32_func_param1, fp32_func_param2)

//LTX >=2
#define field_batchNorm_with_function_deltaXp_v1_k4(stream, LBY, LBX, LTY, LTX, deltaY, Y, N, M, deltaXp1, deltaXp2) \
	field_batchNorm_with_function_deltaXp_v1_kernel_4<LBY, LBX, fp32_func_type>\
		<<< dim3(MIN_int32(M>>LBX>>LTX, GRID_MAX),\
                 MIN_int32(N>>LBY>>LTY, GRID_MAX)),\
            dim3(1<<LBX, 1<<LBY), 0, stream >>>\
			(deltaY, Y, N, M, deltaXp1, deltaXp2, width, stride,\
				fp32_func_param0, fp32_func_param1, fp32_func_param2)

#endif


#ifndef FIELD_BATCH_NORM_WITH_FUNCTION_DELTA_XP_V1_KERNEL_4
#define FIELD_BATCH_NORM_WITH_FUNCTION_DELTA_XP_V1_KERNEL_4

//======[Document]============================================
//<1> N = batch_size =  field_length = lengthv / row_lengthv
//<2> M = row_lengthv
//<3> dims: Y[N, M], X[N, M], X_mean[M], X_mean[M], A[M], B[M]
//<4> dims: deltaY[N, M], deltaX[N, M], deltaXp1[M], deltaXp2[M]
//
//[Forward Propagation]
//(1) Y1 = X_norm = (X - X_mean) * X_rstd
//(1) Y2 = function(Y1) = Y
//
//[Backward Propagation]
//STEP:
//(1) Y1 = function^(-1)(Y2)
//(2) deltaY1 = deltaY2 * derivative_v2(Y1), deltaY2 = deltaY
//(3) X_norm = Y1
//(4) deltaXp2 = field_sum: deltaY1 * X_norm
//(5) deltaXp1 = field_sum: deltaY1
//======[Document]============================================

template<int LBY, int LBX, int fp32_func_type>
__global__ void field_batchNorm_with_function_deltaXp_v1_kernel_4(
	const float* __restrict__ deltaY,
	const float* __restrict__ Y, int N, int M,
	      float* __restrict__ deltaXp1,
	      float* __restrict__ deltaXp2,
	int width, int stride,
	float fp32_func_param0, float fp32_func_param1, float fp32_func_param2)
{
	const int by = blockIdx.y, bx = blockIdx.x;
	const int ty = threadIdx.y, tx = threadIdx.x;

	__shared__ float4 As[1 << LBX][(2 << LBY) + 1];

	//------[parallel field num = 4]----------------------------------------
	const int offsetY = (by << LBY) + ty, stepY = (gridDim.y << LBY);
	const int offsetX = (bx << LBX) + tx, stepX = (gridDim.x << LBX);
	const int stepX4 = stepX << 2, M4 = (M >> 2) << 2;

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

			float4 y1v;//Y1 = function^(-1)(Y2)
			y1v.x = fp32_func_inverse<fp32_func_type>(y2v.x, fp32_func_param0, fp32_func_param1, fp32_func_param2);
			y1v.y = fp32_func_inverse<fp32_func_type>(y2v.y, fp32_func_param0, fp32_func_param1, fp32_func_param2);
			y1v.z = fp32_func_inverse<fp32_func_type>(y2v.z, fp32_func_param0, fp32_func_param1, fp32_func_param2);
			y1v.w = fp32_func_inverse<fp32_func_type>(y2v.w, fp32_func_param0, fp32_func_param1, fp32_func_param2);

			float4 dy1;//deltaY1 = deltaY2 * derivative_v2(Y1)
			dy1.x = dy2.x * fp32_func_derivative_v2<fp32_func_type>(y1v.x, fp32_func_param0, fp32_func_param1, fp32_func_param2);
			dy1.y = dy2.y * fp32_func_derivative_v2<fp32_func_type>(y1v.y, fp32_func_param0, fp32_func_param1, fp32_func_param2);
			dy1.z = dy2.z * fp32_func_derivative_v2<fp32_func_type>(y1v.z, fp32_func_param0, fp32_func_param1, fp32_func_param2);
			dy1.w = dy2.w * fp32_func_derivative_v2<fp32_func_type>(y1v.w, fp32_func_param0, fp32_func_param1, fp32_func_param2);

			float4 dv1;//deltaXp2 = deltaY1 * X_norm
			dv1.x = dy1.x * y1v.x;
			dv1.y = dy1.y * y1v.y;
			dv1.z = dy1.z * y1v.z;
			dv1.w = dy1.w * y1v.w;

			Kahan_simdAdd4(v0, dy1, c0);//deltaXp1 = field_sum: deltaY1
			Kahan_simdAdd4(v1, dv1, c1);//deltaXp2 = field_sum: deltaY1 * X_norm
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

			float2 dXp1 = float2{ sv1.x, sv1.y };//deltaXp1
			float2 dXp2 = float2{ sv1.z, sv1.w };//deltaXp2

			const int xindex2 = x4 + (ty << 1);
			within_width_zero_nan2(dXp1, xindex2, table, stride, width);
			within_width_zero_nan2(dXp2, xindex2, table, stride, width);

			const int dst_index = by * M + xindex2;
			*(float2*)(deltaXp1 + dst_index) = dXp1;//deltaXp1[by, xindex2]
			*(float2*)(deltaXp2 + dst_index) = dXp2;//deltaXp2[by, xindex2]
		}
		__syncthreads();
	}
}

#endif


//======[integration]======================================================
#ifndef FIELD_BATCH_NORM_WITH_FUNCTION_DELTA_XP_V1_STAGE
#define FIELD_BATCH_NORM_WITH_FUNCTION_DELTA_XP_V1_STAGE

//M % 4 == 0, M >= 4
template<int fp32_func_type>
void __temp_field_batchNorm_with_function_deltaXp_v1_stage(cudaStream_t stream,
	const float* deltaY,
	const float* Y, int N, int M,//V1: holdY(), Y is not changed
	      float* deltaXp1,
	      float* deltaXp2,
	int width, int stride,
	float fp32_func_param0, float fp32_func_param1, float fp32_func_param2)
{
	if (M > 15) {
		if (N > 63) { field_batchNorm_with_function_deltaXp_v1_k4(stream, 3, 2, 3, 2, deltaY, Y, N, M, deltaXp1, deltaXp2); return; }//[64, 16]
		if (N > 31) { field_batchNorm_with_function_deltaXp_v1_k4(stream, 3, 2, 2, 2, deltaY, Y, N, M, deltaXp1, deltaXp2); return; }//[32, 16]
		if (N > 15) { field_batchNorm_with_function_deltaXp_v1_k4(stream, 3, 2, 1, 2, deltaY, Y, N, M, deltaXp1, deltaXp2); return; }//[16, 16]
	}
	if (M > 7) {
		if (N > 127) { field_batchNorm_with_function_deltaXp_v1_k4(stream, 4, 1, 3, 2, deltaY, Y, N, M, deltaXp1, deltaXp2); return; }//[ 64, 8]
		if (N >  63) { field_batchNorm_with_function_deltaXp_v1_k4(stream, 4, 1, 2, 2, deltaY, Y, N, M, deltaXp1, deltaXp2); return; }//[ 32, 8]
		if (N >  31) { field_batchNorm_with_function_deltaXp_v1_k4(stream, 4, 1, 1, 2, deltaY, Y, N, M, deltaXp1, deltaXp2); return; }//[ 16, 8]
	}
	field_batchNorm_with_function_deltaXp_v1_k4_small(stream, deltaY, Y, N, M, deltaXp1, deltaXp2);
}


void __field_batchNorm_with_function_deltaXp_v1_stage(JNIEnv *env, cudaStream_t stream,
	const float* deltaY,
	const float* Y, int N, int M,//V1: holdY(), Y is not changed
	      float* deltaXp1,
	      float* deltaXp2,
	int width, int stride,
	int fp32_func_type, float fp32_func_param0, float fp32_func_param1, float fp32_func_param2)
{
	if      (fp32_func_type == FP32_Func_Relu)      __temp_field_batchNorm_with_function_deltaXp_v1_stage<FP32_Func_Relu>     (stream, deltaY, Y, N, M, deltaXp1, deltaXp2, width, stride, fp32_func_param0, fp32_func_param1, fp32_func_param2);
	else if (fp32_func_type == FP32_Func_LeakyRelu) __temp_field_batchNorm_with_function_deltaXp_v1_stage<FP32_Func_LeakyRelu>(stream, deltaY, Y, N, M, deltaXp1, deltaXp2, width, stride, fp32_func_param0, fp32_func_param1, fp32_func_param2);
	else if (fp32_func_type == FP32_Func_Elu)       __temp_field_batchNorm_with_function_deltaXp_v1_stage<FP32_Func_Elu>      (stream, deltaY, Y, N, M, deltaXp1, deltaXp2, width, stride, fp32_func_param0, fp32_func_param1, fp32_func_param2);
	else if (fp32_func_type == FP32_Func_Softplus)  __temp_field_batchNorm_with_function_deltaXp_v1_stage<FP32_Func_Softplus> (stream, deltaY, Y, N, M, deltaXp1, deltaXp2, width, stride, fp32_func_param0, fp32_func_param1, fp32_func_param2);
	else if (fp32_func_type == FP32_Func_Gelu)      __temp_field_batchNorm_with_function_deltaXp_v1_stage<FP32_Func_Gelu>     (stream, deltaY, Y, N, M, deltaXp1, deltaXp2, width, stride, fp32_func_param0, fp32_func_param1, fp32_func_param2);
	else if (fp32_func_type == FP32_Func_Sigmoid)   __temp_field_batchNorm_with_function_deltaXp_v1_stage<FP32_Func_Sigmoid>  (stream, deltaY, Y, N, M, deltaXp1, deltaXp2, width, stride, fp32_func_param0, fp32_func_param1, fp32_func_param2);
	else if (fp32_func_type == FP32_Func_Tanh)      __temp_field_batchNorm_with_function_deltaXp_v1_stage<FP32_Func_Tanh>     (stream, deltaY, Y, N, M, deltaXp1, deltaXp2, width, stride, fp32_func_param0, fp32_func_param1, fp32_func_param2);
	else throwException(env, "Unknown FP32_Func_Type");
}

#endif 


#ifndef FIELD_BATCH_NORM_WITH_FUNCTION_DELTA_XP_V1
#define FIELD_BATCH_NORM_WITH_FUNCTION_DELTA_XP_V1

int __field_batchNorm_with_function_deltaXp_v1(JNIEnv *env,
	cudaStream_t stream1, cudaStream_t stream2,
	const float* deltaY,
	const float* Y, int N, int M,//V1: holdY(), Y is not changed
	      float* deltaXp1_buf, float* deltaXp1,
	      float* deltaXp2_buf, float* deltaXp2,
	int width, int stride,
	int partNum,
	int fp32_func_type, float fp32_func_param0, float fp32_func_param1, float fp32_func_param2)
{
	int nextN = field_nextN(N, M);//N <= HV
	if (nextN <= partNum) {//V.address = NULL, only 1 stage, write result to Y
		__field_batchNorm_with_function_deltaXp_v1_stage(env, stream1,
			deltaY, Y, N, M,
			deltaXp1,
			deltaXp2,
			width, stride,
			fp32_func_type, fp32_func_param0, fp32_func_param1, fp32_func_param2);
		return nextN;
	}

	//V.address != NULL, at least 2 stages
	__field_batchNorm_with_function_deltaXp_v1_stage(env, stream1,
		deltaY, Y, N, M,
		deltaXp1_buf,
		deltaXp2_buf,
		width, stride,
		fp32_func_type, fp32_func_param0, fp32_func_param1, fp32_func_param2);

	//====stream2 shold wait stream1======================================================
	cudaEvent_t event; cudaError_t error;
	error = cudaEventCreate(&event, cudaEventDisableTiming); handleError(error);
	error = cudaEventRecord(event, stream1); handleError(error);
	error = cudaStreamWaitEvent(stream2, event, cudaEventWaitDefault); handleError(error);
	//====stream2 shold wait stream1======================================================

	N = nextN, nextN = field_nextN(N, M);//N % 4 == 0, N >= 4
	while (nextN > partNum) {//end: nextN <= partNum
		__field_sum_stage(stream1, deltaXp1_buf, N, M, deltaXp1_buf, width, stride);
		__field_sum_stage(stream2, deltaXp2_buf, N, M, deltaXp2_buf, width, stride);
		N = nextN, nextN = field_nextN(nextN, M);
	}
	__field_sum_stage(stream1, deltaXp1_buf, N, M, deltaXp1, width, stride);//the last stage
	__field_sum_stage(stream2, deltaXp2_buf, N, M, deltaXp2, width, stride);//the last stage

	error = cudaEventDestroy(event); handleError(error);
	return nextN;
}

#endif

#endif