#pragma once

#ifndef DECONV_3D_DELTAX_ZERO_PADDING_H
#define DECONV_3D_DELTAX_ZERO_PADDING_H

//[ZeroPadding]
#include "dconv3D_dX_ZeroPadding_kernel_s1.cuh"
#include "dconv3D_dX_ZeroPadding_kernel_s1_EX.cuh"
#include "dconv3D_dX_ZeroPadding_kernel_s1_texture.cuh"
#include "dconv3D_dX_ZeroPadding_kernel_A_s1.cuh"
#include "dconv3D_dX_ZeroPadding_kernel_A_s1_texture.cuh"
#include "dconv3D_dX_ZeroPadding_uernel_s1.cuh"
#include "dconv3D_dX_ZeroPadding_uernel_s1_ruse.cuh"

#include "dconv3D_dX_ZeroPadding_kernel_W1.cuh"

//[ZeroPaddingV2]
#include "dconv3D_dX_ZeroPaddingV2_kernel_s1.cuh"
#include "dconv3D_dX_ZeroPaddingV2_kernel_s1_EX.cuh"
#include "dconv3D_dX_ZeroPaddingV2_kernel_s1_EX2.cuh"
#include "dconv3D_dX_ZeroPaddingV2_uernel_s1.cuh"


//======[Integration: 8*8 elements]====================================
#ifndef COMPILE
#define DECONV3D_DX_ZERO_PADDING_UERNEL_S1_8X8
#endif
#ifndef DECONV3D_DX_ZERO_PADDING_UERNEL_S1_8X8
#define DECONV3D_DX_ZERO_PADDING_UERNEL_S1_8X8

template<int LB>
inline bool deconv3D_dX_ZeroPadding_s1_uernel_8x8(cudaStream_t stream, int ic_index, int j_index,
	const float* deltaY, int OH, int OW,
	const float*      W, int FH, int FW,
	      float* deltaX, int IH, int IW,
	int N, int IC, int OC,
	int ph, int pw, 
	int GN, int GM)
{
	bool flag = OC & ((1 << LB) - 1); if (flag) return false;//LB = 4, OC % 16 == 0; LB = 3, OC % 8 == 0

	if (!(GM & 127) && (j_index == 0) && !(N & 7)) {//Kernel A
		if (IS_POWER2(OC)) {
			if (FH == 3 && FW == 3) { u88As1W3_oc2pow(stream, LB, ic_index, j_index, deltaY, OH, OW, W, deltaX, IH, IW, N, IC, LOG2(OC), ph, pw, GN, GM); return true; }
			if (FH == 5 && FW == 5) { u88As1W5_oc2pow(stream, LB, ic_index, j_index, deltaY, OH, OW, W, deltaX, IH, IW, N, IC, LOG2(OC), ph, pw, GN, GM); return true; }
			u88As1_oc2pow(stream, LB, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, LOG2(OC), ph, pw, GN, GM); return true;
		}
		if (FH == 3 && FW == 3) { u88As1W3(stream, LB, ic_index, j_index, deltaY, OH, OW, W, deltaX, IH, IW, N, IC, OC, ph, pw, GN, GM); return true; }
		if (FH == 5 && FW == 5) { u88As1W3(stream, LB, ic_index, j_index, deltaY, OH, OW, W, deltaX, IH, IW, N, IC, OC, ph, pw, GN, GM); return true; }
		u88As1(stream, LB, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, OC, ph, pw, GN, GM); return true;
	}
	if (!(IH & 3) && !(IW & 3) && IS_POWER2(OC)) {
		if (FH == 3 && FW == 3) { u88s1W3x4_oc2pow(stream, LB, ic_index, j_index, deltaY, OH, OW, W, deltaX, IH, IW, IC, LOG2(OC), ph, pw, GN, GM); return true; }
		if (FH == 5 && FW == 5) { u88s1W5x4_oc2pow(stream, LB, ic_index, j_index, deltaY, OH, OW, W, deltaX, IH, IW, IC, LOG2(OC), ph, pw, GN, GM); return true; }
	}
	u88s1(stream, LB, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, ph, pw, GN, GM); return true;
}

#endif



#ifndef COMPILE
#define DECONV_3D_DX_ZERO_PADDING_UERNEL_S1_RUSE_8X8
#endif
#ifndef DECONV_3D_DX_ZERO_PADDING_UERNEL_S1_RUSE_8X8
#define DECONV_3D_DX_ZERO_PADDING_UERNEL_S1_RUSE_8X8

template<int LB>
inline bool deconv3D_dX_ZeroPadding_s1_uernel_ruse_8x8(cudaStream_t stream, int ic_index, int j_index,
	const float* deltaY, int OH, int OW,
	const float*      W, int FH, int FW,
	      float* deltaX, int IH, int IW,
	int IC, int OC,
	int ph, int pw,//sh = sw = 1
	int GN, int GM)
{
	bool flag = OC & ((1 << LB) - 1); if (flag) return false;//LB = 4, OC % 16 == 0; LB = 3, OC % 8 == 0
	if (!(IW & 3) && (FH == FW)) {
		//FH = FW = 2x + 1
		if (FW == 3) { u88s1W3x4_ruse(stream, LB, ic_index, j_index, deltaY, OH, OW, W, deltaX, IH, IW, IC, OC, ph, pw, GN, GM); return true; }
		if (FW == 5) { u88s1x4_ruse(stream, LB, ic_index, j_index, deltaY, OH, OW, W, 5, 5, deltaX, IH, IW, IC, OC, ph, pw, GN, GM); return true; }
		if (FW == 7) { u88s1x4_ruse(stream, LB, ic_index, j_index, deltaY, OH, OW, W, 7, 7, deltaX, IH, IW, IC, OC, ph, pw, GN, GM); return true; }
		if (FW == 9) { u88s1x4_ruse(stream, LB, ic_index, j_index, deltaY, OH, OW, W, 9, 9, deltaX, IH, IW, IC, OC, ph, pw, GN, GM); return true; }
		//FH = FW = 2x
		if (FW == 2) { u88s1x4_ruse(stream, LB, ic_index, j_index, deltaY, OH, OW, W, 2, 2, deltaX, IH, IW, IC, OC, ph, pw, GN, GM); return true; }
		if (FW == 4) { u88s1x4_ruse(stream, LB, ic_index, j_index, deltaY, OH, OW, W, 4, 4, deltaX, IH, IW, IC, OC, ph, pw, GN, GM); return true; }
		if (FW == 6) { u88s1x4_ruse(stream, LB, ic_index, j_index, deltaY, OH, OW, W, 6, 6, deltaX, IH, IW, IC, OC, ph, pw, GN, GM); return true; }
		if (FW == 8) { u88s1x4_ruse(stream, LB, ic_index, j_index, deltaY, OH, OW, W, 8, 8, deltaX, IH, IW, IC, OC, ph, pw, GN, GM); return true; }
	}
	return false;
}

#endif

#endif