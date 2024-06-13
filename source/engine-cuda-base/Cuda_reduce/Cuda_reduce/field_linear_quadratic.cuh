#pragma once

#ifndef FIELD_LINEAR_QUADRATIC_H
#define FIELD_LINEAR_QUADRATIC_H

//(1) field_num % A.width ==0
//(2) field_num -> field_stride = field_num / width * stride
//(3) A.length % feature_num == 0
//(4) A.lengthv % fieled_stride == 0
//(5) row_num = A.lengthv / field_stride
//(6) [y, x] -> [row_num, field_stride]
//(7) [N, M] = [row_num, field_stride], A.lengthv = N*M
//(8) For BN.deltaX: V1: holdY(), Y is not changed \\ affine = false
#ifndef FIELD_LINEAR_QUADRATIC_CALL
#define FIELD_LINEAR_QUADRATIC_CALL

//[16, 4]
#define field_linear_quadratic4_small(stream, A, alpha1, beta1, alpha2, beta2, gamma2, N, M, V1, V2, width, stride) \
	field_linear_quadratic_kernel_4<3, 3>\
		<<< dim3(MIN_int32((M + 31)>>5, GRID_MAX),\
                 MIN_int32((N + 63)>>6, GRID_MAX)),\
            dim3(8, 8), 0, stream>>>\
			(A, alpha1, beta1, alpha2, beta2, gamma2, N, M, V1, V2, width, stride)

//LTX >=2
#define field_linear_quadratic4(stream, LBY, LBX, LTY, LTX, A, alpha1, beta1, alpha2, beta2, gamma2, N, M, V1, V2, width, stride) \
	field_linear_quadratic_kernel_4<LBY, LBX>\
		<<< dim3(MIN_int32(M>>LBX>>LTX, GRID_MAX),\
                 MIN_int32(N>>LBY>>LTY, GRID_MAX)),\
            dim3(1<<LBX, 1<<LBY), 0, stream >>>\
			(A, alpha1, beta1, alpha2, beta2, gamma2, N, M, V1, V2, width, stride)

//gamma2 = beta1 == 0
#define field_linear_quadratic4_add0(stream, LBY, LBX, LTY, LTX, A, alpha1, alpha2, beta2, N, M, V1, V2, width, stride) \
	field_linear_quadratic_kernel_4_add0<LBY, LBX>\
		<<< dim3(MIN_int32(M>>LBX>>LTX, GRID_MAX),\
                 MIN_int32(N>>LBY>>LTY, GRID_MAX)),\
            dim3(1<<LBX, 1<<LBY), 0, stream >>>\
			(A, alpha1, alpha2, beta2, N, M, V1, V2, width, stride)

#endif


#ifndef FIELD_LINEAR_QUADRATIC_KERNEL_4_SHFL
#define FIELD_LINEAR_QUADRATIC_KERNEL_4_SHFL

//warp_shfl: LTX >= 3
#define field_linear_quadratic4_shfl(stream, LBX, LBY, LTX, LTY, A, alpha1, beta1, alpha2, beta2, gamma2, N, M, V1, V2, width, stride) \
	field_linear_quadratic_kernel_4_shfl<LBX, LBY>\
		<<< dim3(N>>LBX>>LTX, M>>LBY>>LTY), dim3(1<<LBX, 1<<LBY), 0, stream >>>\
			(A, alpha1, beta1, alpha2, beta2, gamma2, N, M, V1, V2, width, stride)

template<int LBX, int LBY>
__global__ void field_linear_quadratic_kernel_4_shfl(
	const float* __restrict__ A,
	float alpha1, float beta1,
	float alpha2, float beta2, float gamma2,
	int N, int M,
	float* __restrict__ V1,
	float* __restrict__ V2,
	int width, int stride)
{
	const int bx = blockIdx.x, by = blockIdx.y;
	const int tx = threadIdx.x, ty = threadIdx.y;

	//parallel field num = 4
	const int offsetY = (bx << LBX) + tx, stepY = (gridDim.x << LBX);
	const int offsetX = (by << LBY) + ty, stepX = (gridDim.y << LBY);
	const int stepX4 = stepX << 2, M4 = (M >> 2) << 2;
	for (int x4 = offsetX << 2; x4 < M4; x4 += stepX4)
	{
		float4 v0 = F32_4_0, c0 = F32_4_0;
		float4 v1 = F32_4_0, c1 = F32_4_0;
		for (int y = offsetY; y < N; y += stepY)
		{
			const int aoffset = y * M + x4;
			float4 a = *(float4*)(A + aoffset);//A[y, x4]
			float4 dv0; simdLinear4(dv0, alpha1, a, beta1);//dv0 = alpha1*A + beta1
			float4 dv1; simdQuadratic4(dv1, alpha2, a, beta2, gamma2);//dv1 = alpha2*A^2 + beta2*A + gamma

			Kahan_simdAdd4(v0, dv0, c0);//v1 = v1 + dv1
			Kahan_simdAdd4(v1, dv1, c1);//v2 = v2 + dv2
		}
		__syncthreads();

		float4 t0, t1;
		if (LBX >= 5) {//BLOCK_X == 32
			__SIMD4_shfl_down_sync(t0, 0xffffffff, v0, 16);
			__SIMD4_shfl_down_sync(t1, 0xffffffff, v1, 16);
			Kahan_simdAdd4(v0, t0, c0);//v0 += t0
			Kahan_simdAdd4(v1, t1, c1);//v1 += t1
		}

		if (LBX >= 4) {//BLOCK_X == 16
			__SIMD4_shfl_down_sync(t0, 0xffffffff, v0, 8);
			__SIMD4_shfl_down_sync(t1, 0xffffffff, v1, 8);
			Kahan_simdAdd4(v0, t0, c0);//v0 += t0
			Kahan_simdAdd4(v1, t1, c1);//v1 += t1
		}

		//within a warp(BLOCK_X >= 8)--------------------------------
		__SIMD4_shfl_down_sync(t0, 0xffffffff, v0, 4);//8
		__SIMD4_shfl_down_sync(t1, 0xffffffff, v1, 4);
		Kahan_simdAdd4(v0, t0, c0);//v0 += t0
		Kahan_simdAdd4(v1, t1, c1);//v1 += t1

		__SIMD4_shfl_down_sync(t0, 0xffffffff, v0, 2);//4
		__SIMD4_shfl_down_sync(t1, 0xffffffff, v1, 2);
		Kahan_simdAdd4(v0, t0, c0);//v0 += t0
		Kahan_simdAdd4(v1, t1, c1);//v1 += t1

		__SIMD4_shfl_down_sync(t0, 0xffffffff, v0, 1);//2
		__SIMD4_shfl_down_sync(t1, 0xffffffff, v1, 1);
		Kahan_simdAdd4(v0, t0, c0);//v0 += t0
		Kahan_simdAdd4(v1, t1, c1);//v1 += t1

		if (tx == 0) {
			const int xindex4 = x4 + (tx << 2);
			bool wrt0 = ((xindex4) % stride) < width;
			bool wrt1 = ((xindex4 + 1) % stride) < width;
			bool wrt2 = ((xindex4 + 2) % stride) < width;
			bool wrt3 = ((xindex4 + 3) % stride) < width;
			v0.x *= wrt0; v0.y *= wrt1; v0.z *= wrt2; v0.w *= wrt3;//within_with4
			v1.x *= wrt0; v1.y *= wrt1; v1.z *= wrt2; v1.w *= wrt3;

			const int dst_index = bx * M + xindex4;//[by, xindex4]
			*(float4*)(V1 + dst_index) = v0;
			*(float4*)(V2 + dst_index) = v1;
		}
	}
}

#endif


#ifndef FIELD_LINEAR_QUADRATIC_KERNEL_4
#define FIELD_LINERA_QUADRATIC_KERNEL_4

//[1] A belons to Mat[N, M]
//[2] V1[M] = field_sum: alpha1 * A + beta1,
//[3] V2[M] = field_sum: alpha2 * A^2 + beta2* + gamma2 
template<int LBY, int LBX>
__global__ void field_linear_quadratic_kernel_4(
	const float* __restrict__ A,
	float alpha1, float beta1,
	float alpha2, float beta2, float gamma2,
	int N, int M,
	float* __restrict__ V1,
	float* __restrict__ V2,
	int width, int stride)
{
	const int by = blockIdx.y, bx = blockIdx.x;
	const int ty = threadIdx.y, tx = threadIdx.x;

	__shared__ float4 As[1 << LBX][(2 << LBY) + 1];

	//------[parallel field num = 4]----------------------------------------
	const int offsetY = (by << LBY) + ty, stepY = (gridDim.y << LBY);
	const int offsetX = (bx << LBX) + tx, stepX = (gridDim.x << LBX);
	const int stepX4 = stepX << 2, M4 = (M >> 2) << 2;
	for (int x4 = offsetX << 2; x4 < M4; x4 += stepX4)
	{
		//------compute area[thread reduce: 4 local result]-----------------
		float4 v0 = F32_4_0, c0 = F32_4_0;//v0 = field_sum: A
		float4 v1 = F32_4_0, c1 = F32_4_0;//v1 = field_sum: A^2
		int count = 0;
		for (int y = offsetY; y < N; y += stepY)
		{
			float4 a = *(float4*)(A + y * M + x4);//A[y, x4]
		
			float4 dv1;//v2 = v2 + a^2
			dv1.x = a.x * a.x;
			dv1.y = a.y * a.y;
			dv1.z = a.z * a.z;
			dv1.w = a.w * a.w ;
			
			Kahan_simdAdd4(v0, a, c0);//v1 = v1 + a
			Kahan_simdAdd4(v1, dv1, c1);
			count++;
		}
		
		float Ngamma2 = count * gamma2;//dv1 = alpha2*A^2 + beta2*A + gamma2
		v1.x = alpha2 * v1.x + beta2 * v0.x + Ngamma2;
		v1.y = alpha2 * v1.y + beta2 * v0.y + Ngamma2;
		v1.z = alpha2 * v1.z + beta2 * v0.z + Ngamma2;
		v1.w = alpha2 * v1.w + beta2 * v0.w + Ngamma2;
		c1.x = alpha2 * c1.x + beta2 * c0.x;
		c1.y = alpha2 * c1.y + beta2 * c0.y;
		c1.z = alpha2 * c1.z + beta2 * c0.z;
		c1.w = alpha2 * c1.w + beta2 * c0.w;

		float Nbeta1 = count * beta1; //dv0 = alpha1*A + beta1
		v0.x = alpha1 * v0.x + Nbeta1;
		v0.y = alpha1 * v0.y + Nbeta1;
		v0.z = alpha1 * v0.z + Nbeta1;
		v0.w = alpha1 * v0.w + Nbeta1;
		c0.x = alpha1 * c0.x;
		c0.y = alpha1 * c0.y;
		c0.z = alpha1 * c0.z;
		c0.w = alpha1 * c0.w;

		As[tx][(ty << 1)    ] = float4{ v0.x, v0.y, v1.x, v1.y };
		As[tx][(ty << 1) + 1] = float4{ v0.z, v0.w, v1.z, v1.w };
		__syncthreads();

		//------compute area[block reduce: 4 global result]------------------
		int yIdx;
		float4 scs[2]{//overlap the error in same field
			float4{ c0.x, c0.y, c1.x, c1.y },//yIdx % 2 == 0
			float4{ c0.z, c0.w, c1.z, c1.w } //yIdx % 2 != 0
		};

		if (LBY >= 6) {//BLOCKY = 64
			yIdx = ((ty & 31) << 1) + (ty >> 5);
			if (ty < 64) {//128 -> 64
				float4 sv1 = As[tx][yIdx], sv2 = As[tx][yIdx + 64], sc = scs[yIdx & 1];
				Kahan_simdAdd4(sv1, sv2, sc); As[tx][yIdx] = sv1;
			}
			__syncthreads();
		}

		if (LBY >= 5) {//BLOCKY = 32
			yIdx = ((ty & 15) << 1) + (ty >> 4);
			if (ty < 32) {//64 -> 32
				float4 sv1 = As[tx][yIdx], sv2 = As[tx][yIdx + 32], sc = scs[yIdx & 1];
				Kahan_simdAdd4(sv1, sv2, sc); As[tx][yIdx] = sv1;
			}
			__syncthreads();
		}

		if (LBY >= 4) {//BLOCKY = 16
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
			float4 sv1 = As[tx][yIdx], sv2 = As[tx][yIdx +2], sc = scs[yIdx & 1];
			Kahan_simdAdd4(sv1, sv2, sc);
		
			float2 RV1 = float2{ sv1.x, sv1.y };
			float2 RV2 = float2{ sv1.z, sv1.w };

			const int xindex2 = x4 + (ty << 1);
			bool wrt0 = ((xindex2    ) % stride) < width;
			bool wrt1 = ((xindex2 + 1) % stride) < width;
			RV1.x *= wrt0; RV1.y *= wrt1;//within_width2(RV1, xindex2, stride, width);
			RV2.x *= wrt0; RV2.y *= wrt1;//within_width2(RV2, xindex2, stride, width);

			const int dst_index = by * M + xindex2;//[by, xindex2]
			*(float2*)(V1 + dst_index) = RV1;//deltaXp1 = deltaB
			*(float2*)(V2 + dst_index) = RV2;//deltaXp2 = deltaA
		}
		__syncthreads();
	}
}

#endif


//beta1 = gamma2 = 0
#ifndef FIELD_LINEAR_QUADRATIC_KERNEL_4_ADD0
#define FIELD_LINERA_QUADRATIC_KERNEL_4_ADD0

//[1] A belons to Mat[N, M]
//[2] V1[M] = field_sum: alpha1 * A + beta1,
//[3] V2[M] = field_sum: alpha2 * A^2 + beta2* + gamma2 

template<int LBY, int LBX>
__global__ void field_linear_quadratic_kernel_4_add0(
	const float* __restrict__ A,
	float alpha1,
	float alpha2, float beta2,
	int N, int M,
	float* __restrict__ V1,
	float* __restrict__ V2,
	int width, int stride)
{
	const int by = blockIdx.y, bx = blockIdx.x;
	const int ty = threadIdx.y, tx = threadIdx.x;

	__shared__ float4 As[1 << LBX][(2 << LBY) + 1];

	//------[parallel field num = 4]----------------------------------------
	const int offsetY = (by << LBY) + ty, stepY = (gridDim.y << LBY);
	const int offsetX = (bx << LBX) + tx, stepX = (gridDim.x << LBX);
	const int stepX4 = stepX << 2, M4 = (M >> 2) << 2;
	for (int x4 = offsetX << 2; x4 < M4; x4 += stepX4)
	{
		//------compute area[thread reduce: 4 local result]-----------------
		float4 v0 = F32_4_0, c0 = F32_4_0;//v0 = field_sum: A
		float4 v1 = F32_4_0, c1 = F32_4_0;//v1 = field_sum: A^2
		for (int y = offsetY; y < N; y += stepY)
		{
			const float4 a = *(float4*)(A + y * M + x4);//A[y, x4]

			float4 dv1;//v2 = v2 + a^2
			dv1.x = a.x * a.x;
			dv1.y = a.y * a.y;
			dv1.z = a.z * a.z;
			dv1.w = a.w * a.w;

			Kahan_simdAdd4(v0, a, c0);//v1 = v1 + a
			Kahan_simdAdd4(v1, dv1, c1);//v2 = v2 + a^2
		}
		As[tx][(ty << 1)    ] = float4{ v0.x, v0.y, v1.x, v1.y };
		As[tx][(ty << 1) + 1] = float4{ v0.z, v0.w, v1.z, v1.w };
		__syncthreads();

		//------compute area[block reduce: 4 global result]------------------
		int yIdx;
		float4 scs[2]{//overlap the error in same field
			float4{ c0.x, c0.y, c1.x, c1.y },//yIdx % 2 == 0
			float4{ c0.z, c0.w, c1.z, c1.w } //yIdx % 2 != 0
		};

		if (LBY >= 6) {//BLOCKY = 64
			yIdx = ((ty & 31) << 1) + (ty >> 5);
			if (ty < 64) {//128 -> 64
				float4 sv1 = As[tx][yIdx], sv2 = As[tx][yIdx + 64], sc = scs[yIdx & 1];
				Kahan_simdAdd4(sv1, sv2, sc); As[tx][yIdx] = sv1;
			}
			__syncthreads();
		}

		if (LBY >= 5) {//BLOCKY = 32
			yIdx = ((ty & 15) << 1) + (ty >> 4);
			if (ty < 32) {//64 -> 32
				float4 sv1 = As[tx][yIdx], sv2 = As[tx][yIdx + 32], sc = scs[yIdx & 1];
				Kahan_simdAdd4(sv1, sv2, sc); As[tx][yIdx] = sv1;
			}
			__syncthreads();
		}

		if (LBY >= 4) {//BLOCKY = 16
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

			float2 RV1 = float2{ sv1.x, sv1.y };
			float2 RV2 = float2{ sv1.z, sv1.w };

			RV2.x = alpha2 * RV2.x + beta2 * RV1.x;//V2 = alpha2*X^2 + beta2*X
			RV2.y = alpha2 * RV2.y + beta2 * RV1.y;

			RV1.x = alpha1 * RV1.x;//V1 = alpha1*X
			RV1.y = alpha1 * RV1.y;

			const int xindex2 = x4 + (ty << 1);
			bool wrt0 = ((xindex2    ) % stride) < width;
			bool wrt1 = ((xindex2 + 1) % stride) < width;
			RV1.x *= wrt0; RV1.y *= wrt1;//within_width2(RV1, xindex2, stride, width);
			RV2.x *= wrt0; RV2.y *= wrt1;//within_width2(RV2, xindex2, stride, width);

			const int dst_index = by * M + xindex2;//[by, xindex2]
			*(float2*)(V1 + dst_index) = RV1;//deltaXp1 = deltaB
			*(float2*)(V2 + dst_index) = RV2;//deltaXp2 = deltaA
		}
		__syncthreads();
	}
}

#endif


//======[integration]========================================
#ifndef FIELD_LINEAR_QUADRATIC_STAGE
#define FIELD_LINEAR_QUADRATIC_STAGE

#define FLQ_add0 ((beta1) == 0.0f && (gamma2) == 0.0f)

//M % 4 == 0, M >= 4
void __field_linear_quadratic_stage(cudaStream_t stream,
	const float* A,
	float alpha1, float beta1,
	float alpha2, float beta2, float gamma2,
	int N, int M,
	float* V1, 
	float* V2,
	int width, int stride)
{
	if (M > 15) {
		if (N > 63) {//[64, 16]
			if (FLQ_add0) { field_linear_quadratic4_add0(stream, 3, 2, 3, 2, A, alpha1, alpha2, beta2, N, M, V1, V2, width, stride); return; }
			field_linear_quadratic4(stream, 3, 2, 3, 2, A, alpha1, beta1, alpha2, beta2, gamma2, N, M, V1, V2, width, stride); return; 
		}
		if (N > 31) {//[32, 16]
			if (FLQ_add0) { field_linear_quadratic4_add0(stream, 3, 2, 2, 2, A, alpha1, alpha2, beta2, N, M, V1, V2, width, stride); return; }
			field_linear_quadratic4(stream, 3, 2, 2, 2, A, alpha1, beta1, alpha2, beta2, gamma2, N, M, V1, V2, width, stride); return;
		}
		if (N > 15) {//[16, 16]
			if (FLQ_add0) { field_linear_quadratic4_add0(stream, 3, 2, 1, 2, A, alpha1, alpha2, beta2, N, M, V1, V2, width, stride); return; }
			field_linear_quadratic4(stream, 3, 2, 1, 2, A, alpha1, beta1, alpha2, beta2, gamma2, N, M, V1, V2, width, stride); return; 
		}
	}
	if (M > 7) {
		if (N > 127) {//[ 64, 8]
			if (FLQ_add0) { field_linear_quadratic4_add0(stream, 4, 1, 3, 2, A, alpha1, alpha2, beta2, N, M, V1, V2, width, stride); return; }
			field_linear_quadratic4(stream, 4, 1, 3, 2, A, alpha1, beta1, alpha2, beta2, gamma2, N, M, V1, V2, width, stride); return; 
		}
		if (N >  63) {//[ 32, 8]
			if (FLQ_add0) { field_linear_quadratic4_add0(stream, 4, 1, 2, 2, A, alpha1, alpha2, beta2, N, M, V1, V2, width, stride); return; }
			field_linear_quadratic4(stream, 4, 1, 2, 2, A, alpha1, beta1, alpha2, beta2, gamma2, N, M, V1, V2, width, stride); return; 
		}
		if (N >  31) {//[ 16, 8]
			if (FLQ_add0) { field_linear_quadratic4_add0(stream, 4, 1, 1, 2, A, alpha1, alpha2, beta2, N, M, V1, V2, width, stride); return; }
			field_linear_quadratic4(stream, 4, 1, 1, 2, A, alpha1, beta1, alpha2, beta2, gamma2, N, M, V1, V2, width, stride); return;
		}
	}
	field_linear_quadratic4_small(stream, A, alpha1, beta1, alpha2, beta2, gamma2, N, M, V1, V2, width, stride);
}

#endif 


#ifndef FIELD_LINEAR_QUADRATIC
#define FIELD_LINEAR_QUADRATIC

//new fashion
int __field_linear_quadratic(JNIEnv *env, cudaStream_t stream1, cudaStream_t stream2,
	const float* A,
	float alpha1, float beta1, 
	float alpha2, float beta2, float gamma2,
	int N, int M,
	float* V1, float *Y1,//V1 = Y1.buf
	float* V2, float *Y2,//V2 = Y2.buf
	int width, int stride,
	int partNum)
{
	int nextN = field_nextN(N, M);//N <= HV
	if (nextN <= partNum) {//V.address = NULL, only 1 stage, write result to Y
		__field_linear_quadratic_stage(stream1, A, 
			alpha1, beta1, alpha2, beta2, gamma2, N, M, 
			Y1,//V1 = Y1.buf
			Y2,//V2 = Y2.buf
			width, stride);
		return nextN;
	}

	//V.address != NULL, at least 2 stages
	__field_linear_quadratic_stage(stream1, A, 
		alpha1, beta1, alpha2, beta2, gamma2, N, M, 
		V1,//V1 = Y1.buf
		V2,//V2 = Y2.buf
		width, stride);

	//====stream2 shold wait stream1======================================================
	cudaEvent_t event; cudaError_t error;
	error = cudaEventCreate(&event, cudaEventDisableTiming); handleError(error);
	error = cudaEventRecord(event, stream1); handleError(error);
	error = cudaStreamWaitEvent(stream2, event, cudaEventWaitDefault); handleError(error);
	//====stream2 shold wait stream1======================================================

	N = nextN, nextN = field_nextN(N, M);
	while (nextN > partNum) {//end: nextN <= partNum
		__field_sum_stage(stream1, V1, N, M, V1, width, stride);
		__field_sum_stage(stream2, V2, N, M, V2, width, stride);
		N = nextN, nextN = field_nextN(nextN, M);
	}
	__field_sum_stage(stream1, V1, N, M, Y1, width, stride);//the last stage
	__field_sum_stage(stream2, V2, N, M, Y2, width, stride);//the last stage

	error = cudaEventDestroy(event); handleError(error);
	return nextN;
}

#endif

#endif