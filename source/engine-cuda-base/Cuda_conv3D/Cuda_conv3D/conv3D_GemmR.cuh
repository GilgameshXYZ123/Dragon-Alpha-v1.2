#pragma once

#ifndef CONV_3D_GEMMR_H
#define CONV_3D_GEMMR_H

//[GEMMR]
#include "conv3D_GemmR_kernel.cuh"
#include "conv3D_GemmR_kernel_EX.cuh"
#include "conv3D_GemmR_kernel_EX2.cuh"
#include "conv3D_GemmR_kernel_texture.cuh"
#include "conv3D_GemmR_kernel_texture2.cuh"
#include "conv3D_GemmR_A_kernel.cuh"
#include "conv3D_GemmR_A_kernel_texture.cuh"
#include "conv3D_GemmR_uernel.cuh"
#include "conv3D_GemmR_uernel_EX.cuh"
#include "conv3D_GemmR_uernel_ruse.cuh"
#include "conv3D_GemmR_kernel_W1.cuh"

//[GEMMR V2]
#include "conv3D_GemmV2R_kernel.cuh"
#include "conv3D_GemmV2R_kernel_EX.cuh"
#include "conv3D_GemmV2R_uernel.cuh"


//======[Integration: 8*8 elements]==========================
#ifndef COMPILE
#define CONV_3D_GEMM_UERNEL_S1_8X8R
#endif
#ifndef CONV_3D_GEMM_UERNEL_S1_8X8R
#define CONV_3D_GEMM_UERNEL_S1_8X8R

template<int LB>
inline bool conv3D_Gemm_uernel_8x8R(cudaStream_t stream, int oc_index, int j_index,
	const float*  X, int IH, int IW,
	const float* CW, int FH, int FW,
	float*  Y, int OH, int OW,
	int IC, int OC,
	int sh, int sw, int ph, int pw,
	int GN, int GM)
{
	bool flag = IC & ((1 << LB) - 1); if (flag) return false;//LB = 4, IC % 16 == 0; LB = 3, IC % 8 == 0

	if (IS_POWER2(IC)) {
		if (!(OH & 3) && !(OW & 3)) {
			if (IS_POWER2(FW)) { conv3dGemm_u88R4_fw_ic2pow(stream, LB, oc_index, j_index, X, IH, IW, CW, FH, LOG2(FW), Y, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM); return true; }
			if (FH == 3 && FW == 3) { conv3dGemm_u88R4W3_ic2pow(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM); return true; }
			if (FH == 5 && FW == 5) { conv3dGemm_u88R4W5_ic2pow(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM); return true; }
			if (FH == 7 && FW == 7) { conv3dGemm_u88R4W7_ic2pow(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM); return true; }
			conv3dGemm_u88R4_ic2pow(stream, LB, oc_index, j_index, X, IH, IW, CW, FH, FW, Y, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM); return true;
		}
		if (FH == 3 && FW == 3) { conv3dGemm_u88RW3_ic2pow(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM); return true; }
		if (FH == 5 && FW == 5) { conv3dGemm_u88RW5_ic2pow(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM); return true; }
	}
	if (!(OH & 3) && !(OW & 3)) {
		if (FH == 3 && FW == 3) { conv3dGemm_u88R4W3(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM); return true; }
		if (FH == 5 && FW == 5) { conv3dGemm_u88R4W5(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM); return true; }
		conv3dGemm_u88R4(stream, LB, oc_index, j_index, X, IH, IW, CW, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM); return true;
	}
	if (FH == 3 && FW == 3) { conv3dGemm_u88RW3(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM); return true; }
	if (FH == 5 && FW == 5) { conv3dGemm_u88RW5(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM); return true; }
	conv3dGemm_u88R(stream, LB, oc_index, j_index, X, IH, IW, CW, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM); return true;
}

#endif


#ifndef COMPILE
#define CONV_3D_GEMM_UERNEL_S1_RUSE_8X8R
#endif
#ifndef CONV_3D_GEMM_UERNEL_S1_RUSE_8X8R
#define CONV_3D_GEMM_UERNEL_S1_RUSE_8X8R

template<int LB>
inline bool conv3D_Gemm_uernel_s1_ruse_8x8R(cudaStream_t stream, int oc_index, int j_index,
	const float*  X, int IH, int IW,
	const float* CW, int FH, int FW,
	float*  Y, int OH, int OW,
	int IC, int OC,
	int ph, int pw,//sh = sw = 1
	int GN, int GM)
{
	bool flag = IC & ((1 << LB) - 1); if (flag) return false;//LB = 4, IC % 16 == 0; LB = 3, IC % 8 == 0
	if (!(OW & 3) && (FH == FW)) {//OW % 4 == 0
		//FH = FW = 2x + 1
		if (FW == 3) { conv3dGemm_u88R4W3S1_ruse(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, GN, GM); return true; }
		if (FW == 5) { conv3dGemm_u88R4S1_ruse(stream, LB, oc_index, j_index, X, IH, IW, CW, 5, 5, Y, OH, OW, IC, OC, ph, pw, GN, GM); return true; }
		if (FW == 7) { conv3dGemm_u88R4S1_ruse(stream, LB, oc_index, j_index, X, IH, IW, CW, 7, 7, Y, OH, OW, IC, OC, ph, pw, GN, GM); return true; }
		if (FW == 9) { conv3dGemm_u88R4S1_ruse(stream, LB, oc_index, j_index, X, IH, IW, CW, 9, 9, Y, OH, OW, IC, OC, ph, pw, GN, GM); return true; }
		//FH = FW = 2x
		if (FW == 2) { conv3dGemm_u88R4W2S1_ruse(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, GN, GM); return true; }
		if (FW == 4) { conv3dGemm_u88R4S1_ruse(stream, LB, oc_index, j_index, X, IH, IW, CW, 4, 4, Y, OH, OW, IC, OC, ph, pw, GN, GM); return true; }
		if (FW == 6) { conv3dGemm_u88R4S1_ruse(stream, LB, oc_index, j_index, X, IH, IW, CW, 6, 6, Y, OH, OW, IC, OC, ph, pw, GN, GM); return true; }
		if (FW == 8) { conv3dGemm_u88R4S1_ruse(stream, LB, oc_index, j_index, X, IH, IW, CW, 8, 8, Y, OH, OW, IC, OC, ph, pw, GN, GM); return true; }
	}
	return false;
}


#endif

#endif
