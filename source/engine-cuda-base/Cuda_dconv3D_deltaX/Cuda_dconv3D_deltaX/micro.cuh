#pragma once

#ifndef MICRO_H
#define MICRO_H

inline int MIN_int32(int a, int b) { return (a < b ? a : b); }

inline int LOG2(int n) {
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


#include "micro_dconv3D_dX_idx.cuh"
#include "micro_common.cuh"

#include "micro_ZeroPadding_dense.cuh"
#include "micro_ZeroPadding_V2_dense.cuh"

#include "micro_crossAdd.cuh"

#include "micro_kernelSplit.cuh"
#include "micro_kernelSplit_V2.cuh"
#include "micro_kernelSplit_Ims.cuh"
#include "micro_kernelSplit_Ims2.cuh"

//stat = 4
#include "micro_Winograd_s4_f2x3.cuh"
#include "micro_Winograd_s4_f3x2.cuh"

//stat = 8
#include "micro_Winograd_s8_f7x2.cuh"//FW = 2, out_tile = 7
#include "micro_Winograd_s8_f6x3.cuh"//FW = 3, out_tile = 6,
#include "micro_Winograd_s8_f5x4.cuh"//FW = 4, out_tile = 5,
#include "micro_Winograd_s8_f4x5.cuh"//FW = 5, out_tile = 4
#include "micro_Winograd_s8_f3x6.cuh"//FH = 6, out_tile = 3
#include "micro_Winograd_s8_f2x7.cuh"//FW = 7, out_tile = 2

//stat = 16
#include "micro_Winograd_sg_fAx7.cuh"//FW = 7, out_tile = 10
#include "micro_Winograd_sg_f9x8.cuh"//FW = 8, out_tile =  9
#include "micro_Winograd_sg_f8x9.cuh"//FW = 9, out_tile =  8

#endif