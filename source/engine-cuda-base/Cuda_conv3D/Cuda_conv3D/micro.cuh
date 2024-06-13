#pragma once

#ifndef MICRO_H
#define MICRO_H

//shared_memoryIdx_v1:
//const int idx = (ty << LB) + tx;//threadIdx: 256 = 32 * 8 = 16 * 16
//const int uy = ((idx & 1) + ((idx >> (LB + 1)) << 1)); //((idx & 1) + ((idx / 32) * 2))
//const int ux = ((idx & ((1 << (LB + 1)) - 1)) >> 1);   //(idx % 32) / 2

//shared_memoryIdx_v2:
//const int idx = (ty << LB) + tx;//threadIdx: 256 = 32 * 8 = 16 * 16
//const int vy = idx >> 5, vx = idx & 31;
//const int ux = (vy & 3) * 4 + (vx >> 4) * 2 + (vx & 1); //(vy % 4) * 4 + (vx / 16) * 2 + (vx % 2)
//const int uy = (vy >> 2) * 8 + ((vx & 15) >> 1);        //(vy / 4) * 8 + ((vx % 16) >> 1);

//thread_block swizzcle:
//const int log_tile = 1;
//const int bidx = blockIdx.y * gridDim.x + blockIdx.x;
//const int by = ((bidx / (gridDim.x << log_tile)) << log_tile) + (bidx & ((1 << log_tile) - 1));
//const int bx = (bidx % (gridDim.x << log_tile)) >> log_tile;


inline int MIN_int32(int a, int b) { return (a < b ? a : b); }

int LOG2(int n) {
	int result = 0;
	if (n & 0xffff0000) { result += 16; n >>= 16; }
	if (n & 0x0000ff00) { result += 8; n >>= 8; }
	if (n & 0x000000f0) { result += 4; n >>= 4; }
	if (n & 0x0000000c) { result += 2; n >>= 2; }
	if (n & 0x00000002) { result += 1; n >>= 1; }
	return result;
}

__device__ __forceinline__ void check_zero(float* a, int n) {
	for (int i = 0; i < n; i++)
		if (a[i] == 0) printf("asd: %f\n", a[i]);
}

__device__ __forceinline__ void check_nonzero(float* a, int n) {
	for (int i = 0; i < n; i++)
		if (a[i] != 0) printf("asd: %f\n", a[i]);
}

__device__ __forceinline__ void check_zero_float4(float4 a) {
	if (a.x == 0) printf("asd: %f\n", a.x);
	if (a.y == 0) printf("asd: %f\n", a.y);
	if (a.z == 0) printf("asd: %f\n", a.z);
	if (a.w == 0) printf("asd: %f\n", a.w);
}


#include "micro_conv3D_idx.cuh"
#include "micro_common.cuh"

#include "micro_Gemm.cuh"
#include "micro_GemmV2.cuh"

//stat = 4
#include "micro_Winograd_s4_f3x2.cuh"//FW = 2, out_tile = 3
#include "micro_Winograd_s4_f2x3.cuh"//FW = 3, out_tile = 1

//stat = 8
#include "micro_Winograd_s8_f7x2.cuh"//FW = 2, out_tile = 7
#include "micro_Winograd_s8_f6x3.cuh"//FW = 3, out_tile = 6,
#include "micro_Winograd_s8_f5x4.cuh"//FW = 4, out_tile = 5,
#include "micro_Winograd_s8_f4x5.cuh"//FW = 5, out_tile = 4
#include "micro_Winograd_s8_f3x6.cuh"//FH = 6, out_tile = 3
#include "micro_Winograd_s8_f2x7.cuh"//FW = 7, out_tile = 2

#include "micro_Winograd_f4x3.cuh"

//stat = 16
#include "micro_Winograd_sg_fEx3.cuh"//FW = 3
#include "micro_Winograd_sg_fAx7.cuh"//FW = 7
#include "micro_Winograd_sg_f9x8.cuh"//FW = 8
#include "micro_Winograd_sg_f8x9.cuh"//FW = 9

#endif