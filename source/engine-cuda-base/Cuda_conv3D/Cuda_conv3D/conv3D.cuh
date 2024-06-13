#pragma once

#ifndef CONV3D_H
#define CONV3D_H

#include "conv3D_tensor_remode.cuh"
#include "conv3D_Gemm.cuh"
#include "conv3D_GemmR.cuh"
#include "conv3D_im2col_Winograd.cuh"
#include "conv3D_Winograd2D.cuh"

#include "X.cuh"

#ifdef COMPLIE//<<<<complie-area--------------------------------------------------

//Implicit-GEMM: Common
#ifndef GEMM_AREA
#define GEMM_AREA

#ifndef CONV_3D_GEMM
#define CONV_3D_GEMM

#ifndef CONV_3D_GEMM_MICRO
#define CONV_3D_GEMM_MICRO

#define __conv3D_Gemm(streams, index, length, X, IH, IW, W, FH, FW, Y, OH, OW, N, IC, OC, sh, sw, ph, pw) \
	conv3dGemm(streams, index, length, X,IH,IW, W,FH,FW, Y,OH,OW, N,IC,OC, sh,sw,ph,pw,\
		GET_GN(OC), GET_GM(N, OH, OW), GET_GK(FH, FW, IC), 0, 0)

#define conv3dGemmBranch(SIZE_Y, SIZE_X) {\
	int GNr = GN & (SIZE_Y), GMr = GM & (SIZE_X);\
	if(GNr && GMr){\
		int next_oc_index = (GN - GNr) + oc_index;\
		int next_j_index = (GM - GMr) + j_index;\
		conv3dGemm(streams, index, length, X,IH,IW, W,FH,FW, Y,OH,OW, N,IC,OC, sh,sw,ph,pw,\
			GNr, GM, GK, next_oc_index, j_index);\
		conv3dGemm(streams, index, length, X,IH,IW, W,FH,FW, Y,OH,OW, N,IC,OC, sh,sw,ph,pw,\
            GN, GMr, GK, oc_index, next_j_index);\
		conv3dGemm(streams, index, length, X,IH,IW, W,FH,FW, Y,OH,OW, N,IC,OC, sh,sw,ph,pw,\
            GNr, GMr, GK, next_oc_index, next_j_index);}\
	else if(GNr){\
		int next_oc_index = (GN - GNr) + oc_index;\
		conv3dGemm(streams, index, length, X,IH,IW, W,FH,FW, Y,OH,OW, N,IC,OC, sh,sw,ph,pw,\
			 GNr, GM, GK, next_oc_index, j_index);}\
	else if(GMr){\
		int next_j_index = (GM - GMr) + j_index;\
		conv3dGemm(streams, index, length, X,IH,IW, W,FH,FW, Y,OH,OW, N,IC,OC, sh,sw,ph,pw,\
			 GN, GMr, GK, oc_index, next_j_index);}}

#endif

//(1) GM = N * OH * OW;  GM >= 4, GM % 4 == 0
//(2) GN = OC;           GN >= 4, GN % 4 == 0
//(3) GK = FH * FW * IC; GK >= 8, GK % 4 == 0
//As: FH * FW >= 2, So: GK >= 8
//V2 <=> V1: j_index = ohw_index*N + n_index
void conv3dGemm(jlong* streams, int &index, int length,
	const float* X, int IH, int IW,
	const float* W, int FH, int FW,
	      float* Y, int OH, int OW,
	int N, int IC, int OC,
	int sh, int sw, int ph, int pw,
	int GN, int GM, int GK,
	int oc_index, int j_index)
{
	next_cudaStream(stream, streams, index, length);

	if ((GN > 127) && (GM > 127) && !(GK & 7)) {//[128, 128], GK % 8 == 0
		//======[FW, IC are power of 2]==================================
		if (IS_POWER2(FW) && IS_POWER2(IC)) {
			if (!(OH & 3) && !(OW & 3)) conv3dGemm_k88x4_fw_ic2pow(stream, 4, oc_index, j_index, X, IH, IW, W, FH, LOG2(FW), Y, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);
			else conv3dGemm_k88_fw_ic2pow(stream, 4, oc_index, j_index, X, IH, IW, W, FH, LOG2(FW), Y, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);
		}
		//======[FH = FW = 3]============================================
		else if ((FH == 3) && (FW == 3)) {
			if (IS_POWER2(IC)) {//IC is power of 2
				if (!(OH & 3) && !(OW & 3)) conv3dGemm_k88W3x4_ic2pow(stream, 4, oc_index, j_index, X, IH, IW, W, Y, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);
				else conv3dGemm_k88W3_ic2pow(stream, 4, oc_index, j_index, X, IH, IW, W, Y, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);
			}
			else if (!(OH & 3) && !(OW & 3)) conv3dGemm_k88W3x4(stream, 4, oc_index, j_index, X, IH, IW, W, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
			else conv3dGemm_k88W3(stream, 4, oc_index, j_index, X, IH, IW, W, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
		}
		//======[FH = FW = 5]============================================
		else if ((FH == 5) && (FW == 5)) {
			if (IS_POWER2(IC)) {//IC is power of 2
				if (!(OH & 3) && !(OW & 3)) conv3dGemm_k88W5x4_ic2pow(stream, 4, oc_index, j_index, X, IH, IW, W, Y, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);
				else conv3dGemm_k88W5_ic2pow(stream, 4, oc_index, j_index, X, IH, IW, W, Y, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);
			}
			else if (!(OH & 3) && !(OW & 3)) conv3dGemm_k88W5x4(stream, 4, oc_index, j_index, X, IH, IW, W, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
			else conv3dGemm_k88W5(stream, 4, oc_index, j_index, X, IH, IW, W, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
		}
		//======[Common]=================================================
		else if (IS_POWER2(IC)) {//IC is power of 2
			if (!(OH & 3) && !(OW & 3)) conv3dGemm_k88x4_ic2pow(stream, 4, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);
			else conv3dGemm_k88_ic2pow(stream, 4, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);
		}
		else if (!(OH & 3) && !(OW & 3)) conv3dGemm_k88x4(stream, 4, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
		else conv3dGemm_k88(stream, 4, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
		conv3dGemmBranch(127, 127); return;
	}

	if ((GN > 63) && (GM > 63)) {//[63, 63]
		//======[FW, IC are power of 2]==================================
		if (IS_POWER2(FW) && IS_POWER2(IC)) {
			if (!(OH & 3) && !(OW & 3)) conv3dGemm_k88x4_fw_ic2pow(stream, 3, oc_index, j_index, X, IH, IW, W, FH, LOG2(FW), Y, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);
			else conv3dGemm_k88_fw_ic2pow(stream, 3, oc_index, j_index, X, IH, IW, W, FH, LOG2(FW), Y, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);
		}
		//======[FH = FW = 3]============================================
		else if ((FH == 3) && (FW == 3)) {
			if (IS_POWER2(IC)) {//IC is power of 2
				if (!(OH & 3) && !(OW & 3)) conv3dGemm_k88W3x4_ic2pow(stream, 3, oc_index, j_index, X, IH, IW, W, Y, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);
				else conv3dGemm_k88W3_ic2pow(stream, 3, oc_index, j_index, X, IH, IW, W, Y, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);
			}
			else if (!(OH & 3) && !(OW & 3)) conv3dGemm_k88W3x4(stream, 3, oc_index, j_index, X, IH, IW, W, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
			else conv3dGemm_k88W3(stream, 3, oc_index, j_index, X, IH, IW, W, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
		}
		//======[FH = FW = 5]============================================
		else if ((FH == 5) && (FW == 5)) {
			if (IS_POWER2(IC)) {//IC is power of 2
				if (!(OH & 3) && !(OW & 3)) conv3dGemm_k88W5x4_ic2pow(stream, 3, oc_index, j_index, X, IH, IW, W, Y, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);
				else conv3dGemm_k88W5_ic2pow(stream, 3, oc_index, j_index, X, IH, IW, W, Y, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);
			}
			else if (!(OH & 3) && !(OW & 3)) conv3dGemm_k88W5x4(stream, 3, oc_index, j_index, X, IH, IW, W, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
			else conv3dGemm_k88W5(stream, 3, oc_index, j_index, X, IH, IW, W, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
		}
		//======[Common]=================================================
		else if (IS_POWER2(IC)) {//IC is power of 2
			if (!(OH & 3) && !(OW & 3)) conv3dGemm_k88x4_ic2pow(stream, 3, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);
			else conv3dGemm_k88_ic2pow(stream, 3, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);
		}
		else if (!(OH & 3) && !(OW & 3)) conv3dGemm_k88x4(stream, 3, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
		else conv3dGemm_k88(stream, 3, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
		conv3dGemmBranch(63, 63); return;
	}

	if ((GN > 63) && (GM > 31)) {//[64, 32]
		conv3dPure_k84(stream, 3, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
		conv3dGemmBranch(63, 31); return;
	}
	if ((GN > 31) && (GM > 63)) {//[32, 64]
		conv3dPure_k48(stream, 3, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
		conv3dGemmBranch(31, 63); return;
	}

	if ((GN > 31) && (GM > 31)) {//[32, 32]
		if (!(IC & 3)) conv3dPure_k44(stream, 3, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
		else conv3dGemm_k44(stream, 3, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
		conv3dGemmBranch(31, 31); return;
	}

	if ((GN > 63) && (GM > 15)) {//[64, 16]
		conv3dPure_k82(stream, 3, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
		conv3dGemmBranch(63, 15); return;
	}
	if ((GN > 15) && (GM > 63)) {//[16, 64]
		conv3dPure_k28(stream, 3, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
		conv3dGemmBranch(15, 63); return;
	}

	if ((GN > 31) && (GM > 15)) {//[32, 16]
		conv3dPure_k42(stream, 3, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
		conv3dGemmBranch(31, 15); return;
	}
	if ((GN > 15) && (GM > 31)) {//[16, 32]
		conv3dPure_k24(stream, 3, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
		conv3dGemmBranch(15, 31); return;
	}

	//=======[Small]====================================
	if ((GN > 15) && (GM > 15)) {//[16, 16], GK >= 8
		conv3dGemm_k22(stream, 3, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
		conv3dGemmBranch(15, 15); return;
	}
	if ((GN > 31) && (GM > 7)) {//[32, 8], GK >= 8
		conv3dGemm_k41(stream, 3, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
		conv3dGemmBranch(31, 7); return;
	}
	if ((GN > 15) && (GM > 7)) {//[16, 8], GK >= 8
		conv3dGemm_k21(stream, 3, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
		conv3dGemmBranch(15, 7); return;
	}

	if ((GN > 7) && (GM > 31)) {//[8, 32], GK >= 8
		conv3dGemm_k14(stream, 3, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
		conv3dGemmBranch(7, 31); return;
	}
	if ((GN > 7) && (GM > 15)) {//[8, 16], GK >= 8
		conv3dGemm_k12(stream, 3, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
		conv3dGemmBranch(7, 15); return;
	}
	if ((GN > 7) && (GM > 7)) {//[8, 8], GK >= 8
		conv3dGemm_k11(stream, 3, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
		conv3dGemmBranch(7, 7); return;
	}

	if (GN > 15) {//[16, 4]
		conv3dGemm_k41(stream, 2, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
		conv3dGemmBranch(15, 3); return;
	}
	if (GN > 7) {//[8, 4]
		conv3dGemm_k21(stream, 2, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
		conv3dGemmBranch(7, 3); return;
	}

	if (GM > 31) {//[4, 32], GK >= 8
		conv3dGemm_s1_4x2(stream, 3, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
		conv3dGemmBranch(3, 31); return;
	}
	if (GM > 15) {//[4, 16], GK >= 8
		conv3dGemm_s1_2x2(stream, 3, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
		conv3dGemmBranch(3, 15); return;
	}
	if (GM > 7) {//[4, 8]
		conv3dGemm_k12(stream, 2, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
		conv3dGemmBranch(3, 7); return;
	}

	conv3dGemm_k11(stream, 2, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
}

#endif


#ifndef CONV_3D_GEMM_TEXTURE
#define CONV_3D_GEMM_TEXTURE

#ifndef CONV_3D_GEMM_TEXTURE_MICRO
#define CONV_3D_GEMM_TEXTURE_MICRO

#define __conv3D_Gemm_tex(streams, index, length, texX, X, IH, IW, W, FH, FW, Y, OH, OW, N, IC, OC, sh, sw, ph, pw) \
	conv3dGemm_texture(streams, index, length, texX,X,IH,IW, W,FH,FW, Y,OH,OW, N,IC,OC, sh,sw,ph,pw,\
		GET_GN(OC), GET_GM(N, OH, OW), GET_GK(FH, FW, IC), 0, 0)

#define conv3dGemmBranch_tex(SIZE_Y, SIZE_X) {\
	int GNr = GN & (SIZE_Y), GMr = GM & (SIZE_X);\
	if(GNr && GMr){\
		int next_oc_index = (GN - GNr) + oc_index;\
		int next_j_index = (GM - GMr) + j_index;\
		conv3dGemm_texture(streams, index, length, texX,X,IH,IW, W,FH,FW, Y,OH,OW, N,IC,OC, sh,sw,ph,pw,\
			GNr, GM, GK, next_oc_index, j_index);\
		conv3dGemm_texture(streams, index, length, texX,X,IH,IW, W,FH,FW, Y,OH,OW, N,IC,OC, sh,sw,ph,pw,\
            GN, GMr, GK, oc_index, next_j_index);\
		conv3dGemm_texture(streams, index, length, texX,X,IH,IW, W,FH,FW, Y,OH,OW, N,IC,OC, sh,sw,ph,pw,\
            GNr, GMr, GK, next_oc_index, next_j_index);}\
	else if(GNr){\
		int next_oc_index = (GN - GNr) + oc_index;\
		conv3dGemm_texture(streams, index, length, texX,X,IH,IW, W,FH,FW, Y,OH,OW, N,IC,OC, sh,sw,ph,pw,\
			 GNr, GM, GK, next_oc_index, j_index);}\
	else if(GMr){\
		int next_j_index = (GM - GMr) + j_index;\
		conv3dGemm_texture(streams, index, length, texX,X,IH,IW, W,FH,FW, Y,OH,OW, N,IC,OC, sh,sw,ph,pw,\
			 GN, GMr, GK, oc_index, next_j_index);}}

#endif

//(1) GM = N * OH * OW;  GM >= 4, GM % 4 == 0
//(2) GN = OC;           GN >= 4, GN % 4 == 0
//(3) GK = FH * FW * IC; GK >= 8, GK % 4 == 0
//As: FH * FW >= 2, So: GK >= 8
//V2 <=> V1: j_index = ohw_index*N + n_index
void conv3dGemm_texture(jlong* streams, int &index, int length,
	cudaTextureObject_t texX, const float* X, int IH, int IW,
	const float* W, int FH, int FW,
	      float* Y, int OH, int OW,
	int N, int IC, int OC,
	int sh, int sw, int ph, int pw,
	int GN, int GM, int GK,
	int oc_index, int j_index)
{
	next_cudaStream(stream, streams, index, length);

	if ((GN > 127) && (GM > 127) && !(GK & 7)) {//[128, 128], GK % 8 == 0
		//======[FW, IC are power of 2]==================================
		if (IS_POWER2(FW) && IS_POWER2(IC)) {
			if (!(OH & 3) && !(OW & 3)) conv3dGemm_k88x4_fw_ic2pow_tex(stream, 4, oc_index, j_index, texX, IH, IW, W, FH, LOG2(FW), Y, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);
			else conv3dGemm_k88_fw_ic2pow_tex(stream, 4, oc_index, j_index, texX, IH, IW, W, FH, LOG2(FW), Y, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);
		}
		//======[FH = FW = 3]============================================
		else if ((FH == 3) && (FW == 3)) {
			if (IS_POWER2(IC)) {//IC is power of 2
				if (!(OH & 3) && !(OW & 3)) conv3dGemm_k88W3x4_ic2pow_tex(stream, 4, oc_index, j_index, texX, IH, IW, W, Y, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);
				else conv3dGemm_k88W3_ic2pow_tex(stream, 4, oc_index, j_index, texX, IH, IW, W, Y, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);
			}
			else if (!(OH & 3) && !(OW & 3)) conv3dGemm_k88W3x4_tex(stream, 4, oc_index, j_index, texX, IH, IW, W, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
			else conv3dGemm_k88W3_tex(stream, 4, oc_index, j_index, texX, IH, IW, W, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
		}
		//======[FH = FW = 5]============================================
		else if ((FH == 5) && (FW == 5)) {
			if (IS_POWER2(IC)) {//IC is power of 2
				if (!(OH & 3) && !(OW & 3)) conv3dGemm_k88W5x4_ic2pow_tex(stream, 4, oc_index, j_index, texX, IH, IW, W, Y, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);
				else conv3dGemm_k88W5_ic2pow_tex(stream, 4, oc_index, j_index, texX, IH, IW, W, Y, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);
			}
			else if (!(OH & 3) && !(OW & 3)) conv3dGemm_k88W5x4_tex(stream, 4, oc_index, j_index, texX, IH, IW, W, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
			else conv3dGemm_k88W5_tex(stream, 4, oc_index, j_index, texX, IH, IW, W, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
		}
		//======[Common]=================================================
		else if (IS_POWER2(IC)) {//IC is power of 2
			if (!(OH & 3) && !(OW & 3)) conv3dGemm_k88x4_ic2pow_tex(stream, 4, oc_index, j_index, texX, IH, IW, W, FH, FW, Y, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);
			else conv3dGemm_k88_ic2pow_tex(stream, 4, oc_index, j_index, texX, IH, IW, W, FH, FW, Y, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);
		}
		else if (!(OH & 3) && !(OW & 3)) conv3dGemm_k88x4_tex(stream, 4, oc_index, j_index, texX, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
		else conv3dGemm_k88_tex(stream, 4, oc_index, j_index, texX, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
		conv3dGemmBranch_tex(127, 127); return;
	}

	if ((GN > 63) && (GM > 63)) {//[63, 63]
		//======[FW, IC are power of 2]==================================
		if (IS_POWER2(FW) && IS_POWER2(IC)) {
			if (!(OH & 3) && !(OW & 3)) conv3dGemm_k88x4_fw_ic2pow_tex(stream, 3, oc_index, j_index, texX, IH, IW, W, FH, LOG2(FW), Y, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);
			else conv3dGemm_k88_fw_ic2pow_tex(stream, 3, oc_index, j_index, texX, IH, IW, W, FH, LOG2(FW), Y, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);
		}
		//======[FH = FW = 3]============================================
		else if ((FH == 3) && (FW == 3)) {
			if (IS_POWER2(IC)) {//IC is power of 2
				if (!(OH & 3) && !(OW & 3)) conv3dGemm_k88W3x4_ic2pow_tex(stream, 3, oc_index, j_index, texX, IH, IW, W, Y, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);
				else conv3dGemm_k88W3_ic2pow_tex(stream, 3, oc_index, j_index, texX, IH, IW, W, Y, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);
			}
			else if (!(OH & 3) && !(OW & 3)) conv3dGemm_k88W3x4_tex(stream, 3, oc_index, j_index, texX, IH, IW, W, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
			else conv3dGemm_k88W3_tex(stream, 3, oc_index, j_index, texX, IH, IW, W, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
		}
		//======[FH = FW = 5]============================================
		else if ((FH == 5) && (FW == 5)) {
			if (IS_POWER2(IC)) {//IC is power of 2
				if (!(OH & 3) && !(OW & 3)) conv3dGemm_k88W5x4_ic2pow_tex(stream, 3, oc_index, j_index, texX, IH, IW, W, Y, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);
				else conv3dGemm_k88W5_ic2pow_tex(stream, 3, oc_index, j_index, texX, IH, IW, W, Y, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);
			}
			else if (!(OH & 3) && !(OW & 3)) conv3dGemm_k88W5x4_tex(stream, 3, oc_index, j_index, texX, IH, IW, W, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
			else conv3dGemm_k88W5_tex(stream, 3, oc_index, j_index, texX, IH, IW, W, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
		}
		//======[Common]=================================================
		else if (IS_POWER2(IC)) {//IC is power of 2
			if (!(OH & 3) && !(OW & 3)) conv3dGemm_k88x4_ic2pow_tex(stream, 3, oc_index, j_index, texX, IH, IW, W, FH, FW, Y, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);
			else conv3dGemm_k88_ic2pow_tex(stream, 3, oc_index, j_index, texX, IH, IW, W, FH, FW, Y, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);
		}
		else if (!(OH & 3) && !(OW & 3)) conv3dGemm_k88x4_tex(stream, 3, oc_index, j_index, texX, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
		else conv3dGemm_k88_tex(stream, 3, oc_index, j_index, texX, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
		conv3dGemmBranch_tex(63, 63); return;
	}

	if ((GN > 63) && (GM > 31)) {//[64, 32]
		conv3dPure_k84(stream, 3, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
		conv3dGemmBranch(63, 31); return;
	}
	if ((GN > 31) && (GM > 63)) {//[32, 64]
		conv3dPure_k48_tex(stream, 3, oc_index, j_index, texX, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
		conv3dGemmBranch(31, 63); return;
	}

	if ((GN > 31) && (GM > 31)) {//[32, 32]
		if (!(IC & 3)) conv3dPure_k44(stream, 3, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
		else conv3dGemm_k44(stream, 3, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
		conv3dGemmBranch(31, 31); return;
	}

	if ((GN > 63) && (GM > 15)) {//[64, 16]
		conv3dPure_k82(stream, 3, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
		conv3dGemmBranch(63, 15); return;
	}
	if ((GN > 15) && (GM > 63)) {//[16, 64]
		conv3dPure_k28(stream, 3, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
		conv3dGemmBranch(15, 63); return;
	}

	if ((GN > 31) && (GM > 15)) {//[32, 16]
		conv3dPure_k42(stream, 3, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
		conv3dGemmBranch(31, 15); return;
	}
	if ((GN > 15) && (GM > 31)) {//[16, 32]
		conv3dPure_k24(stream, 3, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
		conv3dGemmBranch(15, 31); return;
	}

	//=======[Small]=================================
	if ((GN > 15) && (GM > 15)) {//[16, 16], GK >= 8
		conv3dGemm_k22(stream, 3, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
		conv3dGemmBranch(15, 15); return;
	}

	if ((GN > 31) && (GM > 7)) {//[32, 8], GK >= 8
		conv3dGemm_k41(stream, 3, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
		conv3dGemmBranch(31, 7); return;
	}
	if ((GN > 15) && (GM > 7)) {//[16, 8], GK >= 8
		conv3dGemm_k21(stream, 3, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
		conv3dGemmBranch(15, 7); return;
	}

	if ((GN > 7) && (GM > 31)) {//[8, 32], GK >= 8
		conv3dGemm_k14(stream, 3, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
		conv3dGemmBranch(7, 31); return;
	}
	if ((GN > 7) && (GM > 15)) {//[8, 16], GK >= 8
		conv3dGemm_k12(stream, 3, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
		conv3dGemmBranch(7, 15); return;
	}
	if ((GN > 7) && (GM > 7)) {//[8, 8], GK >= 8
		conv3dGemm_k11(stream, 3, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
		conv3dGemmBranch(7, 7); return;
	}

	if (GN > 15) {//[16, 4]
		conv3dGemm_k41(stream, 2, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
		conv3dGemmBranch(15, 3); return;
	}
	if (GN > 7) {//[8, 4]
		conv3dGemm_k21(stream, 2, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
		conv3dGemmBranch(7, 3); return;
	}

	if (GM > 31) {//[4, 32], GK >= 8
		conv3dGemm_s1_4x2(stream, 3, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
		conv3dGemmBranch(3, 31); return;
	}
	if (GM > 15) {//[4, 16], GK >= 8
		conv3dGemm_s1_2x2(stream, 3, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
		conv3dGemmBranch(3, 15); return;
	}
	if (GM > 7) {//[4, 8]
		conv3dGemm_k12(stream, 2, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
		conv3dGemmBranch(3, 7); return;
	}
	conv3dGemm_k11(stream, 2, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
}

#endif


#ifndef CONV_3D_GEMMV2
#define CONV_3D_GEMMV2

#define __conv3D_GemmV2(env, streams, index, length, X, IH, IW, sizeX, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw) \
	conv3dGemmV2(env, streams,index,length, X,IH,IW,sizeX, W,FH,FW, Y,OH,OW, IC,OC, sh,sw,ph,pw,\
		GET_GN(OC), N, 0, 0)

//(1) GM = N * OH * OW;  GM >= 4, GM % 4 == 0
//(2) GN = OC;           GN >= 4, GN % 4 == 0
//(3) GK = FH * FW * IC; GK >= 8, GK % 4 == 0
//when: N >= 64
void conv3dGemmV2(JNIEnv *env, jlong* streams, int &index, int length,
	const float* X, int IH, int IW, int sizeX,
	const float* W, int FH, int FW,
	      float* Y, int OH, int OW,
	int IC, int OC,//(GN, GM, bz) = (OC, N, OH*OW)
	int sh, int sw, int ph, int pw,
	int GN, int N,//GM = N != X.N
	int oc_index, int n_index)//j_index = n_index * OH_OW
{
	next_cudaStream(stream, streams, index, length);

	if ((GN > 127) && (N > 127) && !(IC & 7)) {//IC % 8 == 0
		if (IS_POWER2(IC)) {
			if (CAN_V2_W3P1) conv3dGemmV2_k88W3p1_ic2pow(stream, 4, oc_index, n_index, X, IH, IW, W, Y, OH, OW, LOG2(IC), OC, sh, sw, GN, N);
			else if (CAN_V2_W4P1) conv3dGemmV2_k88W4p1_ic2pow(stream, 4, oc_index, n_index, X, IH, IW, W, Y, OH, OW, LOG2(IC), OC, sh, sw, GN, N);
			else if (CAN_V2_W5P2) conv3dGemmV2_k88W5p2_ic2pow(stream, 4, oc_index, n_index, X, IH, IW, W, Y, OH, OW, LOG2(IC), OC, sh, sw, GN, N);
			else conv3dGemmV2_k88(stream, 4, oc_index, n_index, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, N);
		}
		else conv3dGemmV2_k88(stream, 4, oc_index, n_index, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, N);
	}

	V2_TO_V1(FH, FW, OH, OW, N, IC, n_index);//=======================================================
	int GNr = GN & 127, GMr = (N & 127) * OH_OW;
	if (env) {//env != null, use texture
		cudaTextureObject_t texX = floatTexture((float*)X, sizeX, env);
		if (GNr && GMr) {
			int next_oc_index = (GN - GNr) + oc_index;
			int next_j_index = (GM - GMr) + j_index;
			conv3dGemm_texture(streams, index, length, texX, X, IH, IW, W, FH, FW, Y, OH, OW, N, IC, OC, sh, sw, ph, pw,
				GNr, GM, GK, next_oc_index, j_index);
			conv3dGemm_texture(streams, index, length, texX, X, IH, IW, W, FH, FW, Y, OH, OW, N, IC, OC, sh, sw, ph, pw,
				GN, GMr, GK, oc_index, next_j_index);
			conv3dGemm_texture(streams, index, length, texX, X, IH, IW, W, FH, FW, Y, OH, OW, N, IC, OC, sh, sw, ph, pw,
				GNr, GMr, GK, next_oc_index, next_j_index);
		}
		else if (GNr) {
			int next_oc_index = (GN - GNr) + oc_index;
			conv3dGemm_texture(streams, index, length, texX, X, IH, IW, W, FH, FW, Y, OH, OW, N, IC, OC, sh, sw, ph, pw,
				GNr, GM, GK, next_oc_index, j_index);
		}
		else if (GMr) {
			int next_j_index = (GM - GMr) + j_index;
			conv3dGemm_texture(streams, index, length, texX, X, IH, IW, W, FH, FW, Y, OH, OW, N, IC, OC, sh, sw, ph, pw,
				GN, GMr, GK, oc_index, next_j_index);
		}
		cudaError_t error = cudaDestroyTextureObject(texX); handleError(error);
		return;
	}

	if (GNr && GMr) {
		int next_oc_index = (GN - GNr) + oc_index; 
		int next_j_index = (GM - GMr) + j_index; 
		conv3dGemm(streams, index, length, X, IH, IW, W, FH, FW, Y, OH, OW, N, IC, OC, sh, sw, ph, pw,
			GNr, GM, GK, next_oc_index, j_index); 
		conv3dGemm(streams, index, length, X, IH, IW, W, FH, FW, Y, OH, OW, N, IC, OC, sh, sw, ph, pw,
			GN, GMr, GK, oc_index, next_j_index); 
		conv3dGemm(streams, index, length, X, IH, IW, W, FH, FW, Y, OH, OW, N, IC, OC, sh, sw, ph, pw,
			GNr, GMr, GK, next_oc_index, next_j_index);
	}
	else if (GNr) {
		int next_oc_index = (GN - GNr) + oc_index;
		conv3dGemm(streams, index, length, X, IH, IW, W, FH, FW, Y, OH, OW, N, IC, OC, sh, sw, ph, pw, 
			GNr, GM, GK, next_oc_index, j_index);
	}
	else if (GMr) {
		int next_j_index = (GM - GMr) + j_index; 
		conv3dGemm(streams, index, length, X, IH, IW, W, FH, FW, Y, OH, OW, N, IC, OC, sh, sw, ph, pw,
			GN, GMr, GK, oc_index, next_j_index);
	}
}



#endif


#ifndef CONV_3D_GEMM_NO_PADDING
#define CONV_3D_GEMM_NO_PADDING

#ifndef CONV_3D_GEMM_NO_PADDING_MICRO
#define CONV_3D_GEMM_NO_PADDING_MICRO

#define __conv3D_Gemm_np(streams, index, length, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw) \
	conv3dGemm_np(streams, index, length, X,IH,IW, W,FH,FW, Y,OH,OW, IC,OC, sh,sw,\
		GET_GN(OC), GET_GM(N, OH, OW), GET_GK(FH, FW, IC), 0, 0)

#define conv3dGemmNPBranch(SIZE_Y, SIZE_X) {\
	int GNr = GN & (SIZE_Y), GMr = GM & (SIZE_X);\
	if(GNr && GMr){\
		int next_oc_index = (GN - GNr) + oc_index;\
		int next_j_index = (GM - GMr) + j_index;\
		conv3dGemm_np(streams, index, length, X,IH,IW, W,FH,FW, Y,OH,OW, IC,OC, sh,sw,\
			GNr, GM, GK, next_oc_index, j_index);\
		conv3dGemm_np(streams, index, length, X,IH,IW, W,FH,FW, Y,OH,OW, IC,OC, sh,sw,\
            GN, GMr, GK, oc_index, next_j_index);\
		conv3dGemm_np(streams, index, length, X,IH,IW, W,FH,FW, Y,OH,OW, IC,OC, sh,sw,\
            GNr, GMr, GK, next_oc_index, next_j_index);}\
	else if(GNr){\
		int next_oc_index = (GN - GNr) + oc_index;\
		conv3dGemm_np(streams, index, length, X,IH,IW, W,FH,FW, Y,OH,OW, IC,OC, sh,sw,\
			 GNr, GM, GK, next_oc_index, j_index);}\
	else if(GMr){\
		int next_j_index = (GM - GMr) + j_index;\
		conv3dGemm_np(streams, index, length, X,IH,IW, W,FH,FW, Y,OH,OW, IC,OC, sh,sw,\
			 GN, GMr, GK, oc_index, next_j_index);}}

#endif

//(1) GM = N * OH * OW;  GM >= 4, GM % 4 == 0
//(2) GN = OC;           GN >= 4, GN % 4 == 0
//(3) GK = FH * FW * IC; GK >= 8, GK % 4 ==0
//(4) ph = pw = 0
void conv3dGemm_np(jlong* streams, int &index, int length,
	const float* X, int IH, int IW,
	const float* W, int FH, int FW,
	      float* Y, int OH, int OW,
	int IC, int OC,
	int sh, int sw,
	int GN, int GM, int GK,
	int oc_index, int j_index)
{
	next_cudaStream(stream, streams, index, length);

	if (!(OH & 3) && !(OW & 3)) {//(OH, OW) % 4 == 0
		if ((GN > 127) && (GM > 127) && !(GK & 7)) {//GK % 8 == 0
			if (IS_POWER2(IC)) {
				if (IS_POWER2(FW)) conv3dGemm_k88x4_np_fw_ic2pow(stream, 4, oc_index, j_index, X, IH, IW, W, FH, LOG2(FW), Y, OH, OW, LOG2(IC), OC, sh, sw, GN, GM);
				else conv3dGemm_k88x4_np_ic2pow(stream, 4, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, LOG2(IC), OC, sh, sw, GN, GM);
			}
			else conv3dGemm_k88x4_np(stream, 4, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, GN, GM);
			conv3dGemmNPBranch(127, 127); return;
		}

		if ((GN > 63) && (GM > 63)) {
			if (IS_POWER2(IC)) {
				if (IS_POWER2(FW)) conv3dGemm_k88x4_np_fw_ic2pow(stream, 3, oc_index, j_index, X, IH, IW, W, FH, LOG2(FW), Y, OH, OW, LOG2(IC), OC, sh, sw, GN, GM);
				else conv3dGemm_k88x4_np_ic2pow(stream, 3, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, LOG2(IC), OC, sh, sw, GN, GM);
			}
			else conv3dGemm_k88x4_np(stream, 3, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, GN, GM);
			conv3dGemmNPBranch(63, 63); return;
		}
	}

	if ((GN > 127) && (GM > 127) && !(GK & 7)) {//GK % 8 == 0
		if (IS_POWER2(IC)) {
			if (IS_POWER2(FW)) conv3dGemm_k88_np_fw_ic2pow(stream, 4, oc_index, j_index, X, IH, IW, W, FH, LOG2(FW), Y, OH, OW, LOG2(IC), OC, sh, sw, GN, GM);
			else conv3dGemm_k88_np_ic2pow(stream, 4, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, LOG2(IC), OC, sh, sw, GN, GM);
		}
		else conv3dGemm_k88_np(stream, 4, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, GN, GM);
		conv3dGemmNPBranch(127, 127); return;
	}

	if ((GN > 63) && (GM > 63)) {
		if (IS_POWER2(IC)) {
			if (IS_POWER2(FW)) conv3dGemm_k88_np_fw_ic2pow(stream, 3, oc_index, j_index, X, IH, IW, W, FH, LOG2(FW), Y, OH, OW, LOG2(IC), OC, sh, sw, GN, GM);
			else conv3dGemm_k88_np_ic2pow(stream, 3, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, LOG2(IC), OC, sh, sw, GN, GM);
		}
		else conv3dGemm_k88_np(stream, 3, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, GN, GM);
		conv3dGemmNPBranch(63, 63); return;
	}

	if ((GN > 63) && (GM > 31)) {
		if (IS_POWER2(IC)) {
			if (IS_POWER2(FW)) conv3dGemm_k84_np_fw_ic2pow(stream, 3, oc_index, j_index, X, IH, IW, W, FH, LOG2(FW), Y, OH, OW, LOG2(IC), OC, sh, sw, GN, GM);
			else conv3dGemm_k84_np_ic2pow(stream, 3, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, LOG2(IC), OC, sh, sw, GN, GM);
		}
		else conv3dGemm_k84_np(stream, 3, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, GN, GM);
		conv3dGemmNPBranch(63, 31); return;
	}

	if ((GN > 31) && (GM > 63)) {
		if (IS_POWER2(IC) && IS_POWER2(FW)) {
			conv3dGemm_k48_np_fw_ic2pow(stream, 3, oc_index, j_index, X, IH, IW, W, FH, LOG2(FW), Y, OH, OW, LOG2(IC), OC, sh, sw, GN, GM);
		}
		else conv3dGemm_k48_np(stream, 3, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, GN, GM);
		conv3dGemmNPBranch(31, 63); return;
	}

	if ((GN > 31) && (GM > 31)) {
		if (IS_POWER2(IC) && IS_POWER2(FW)) {
			conv3dGemm_k44_np_fw_ic2pow(stream, 3, oc_index, j_index, X, IH, IW, W, FH, LOG2(FW), Y, OH, OW, LOG2(IC), OC, sh, sw, GN, GM);
			conv3dGemmNPBranch(31, 31); return;
		}
		if (IS_POWER2(IC)) {
			conv3dGemm_k44_np_ic2pow(stream, 3, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, LOG2(IC), OC, sh, sw, GN, GM);
			conv3dGemmNPBranch(31, 31); return;
		}
		if (!(IC & 3)) {//IC % 4 == 0
			conv3dPure_k44(stream, 3, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, 0, 0, GN, GM);
			conv3dGemmNPBranch(31, 31); return;
		}
		conv3dGemm_k44(stream, 3, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, 0, 0, GN, GM);
		conv3dGemmNPBranch(31, 31); return;
	}

	if ((GN > 63) && (GM > 15)) {//[64, 16]
		conv3dPure_k82(stream, 3, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, 0, 0, GN, GM);
		conv3dGemmNPBranch(63, 15); return;
	}
	if ((GN > 15) && (GM > 63)) {//[16, 64]
		conv3dPure_k28(stream, 3, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, 0, 0, GN, GM);
		conv3dGemmNPBranch(15, 63); return;
	}

	if ((GN > 31) && (GM > 15)) {//GK % 4 == 0
		conv3dPure_k42(stream, 3, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, 0, 0, GN, GM);
		conv3dGemmNPBranch(31, 15); return;
	}
	if ((GN > 15) && (GM > 31)) {//GK % 4 == 0
		conv3dPure_k24(stream, 3, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, 0, 0, GN, GM);
		conv3dGemmNPBranch(15, 31); return;
	}

	if ((GN > 15) && (GM > 15)) {//GK >= 8
		conv3dGemm_k22_np(stream, 3, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, GN, GM);
		conv3dGemmNPBranch(15, 15); return;
	}
	if ((GN > 31) && (GM > 7)) {//GK >= 8
		conv3dGemm_k41_np(stream, 3, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, GN, GM);
		conv3dGemmNPBranch(31, 7); return;
	}
	if ((GN > 7) && (GM > 31)) {//GK >= 8
		conv3dGemm_k14_np(stream, 3, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, GN, GM);
		conv3dGemmNPBranch(7, 31); return;
	}
	if ((GN > 15) && (GM > 7)) {//GK >= 8
		conv3dGemm_k21(stream, 3, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, 0, 0, GN, GM);
		conv3dGemmNPBranch(15, 7); return;
	}
	if ((GN > 7) && (GM > 15)) {//GK >= 8
		conv3dGemm_k12(stream, 3, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, 0, 0, GN, GM);
		conv3dGemmNPBranch(7, 15); return;
	}
	if ((GN > 7) && (GM > 7)) {//GK >= 8
		conv3dGemm_k11(stream, 3, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, 0, 0, GN, GM);
		conv3dGemmNPBranch(7, 7); return;
	}
	if (GN > 15) {//GK >= 4, GM >= 4
		conv3dGemm_k41_np(stream, 2, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, GN, GM);
		conv3dGemmNPBranch(15, 3); return;
	}

	if (GM > 15) {//GK >= 4, GM >= 4
		conv3dGemm_k14_np(stream, 2, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, GN, GM);
		conv3dGemmNPBranch(3, 15); return;
	}
	if (GN > 7) {//GK >= 4, GM >= 4
		conv3dGemm_k21(stream, 2, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, 0, 0, GN, GM);
		conv3dGemmNPBranch(7, 3); return;
	}
	if (GM > 7) {//GK > 4, GN > 4
		conv3dGemm_k12(stream, 2, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, 0, 0, GN, GM);
		conv3dGemmNPBranch(3, 7); return;
	}
	conv3dGemm_k11(stream, 2, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, 0, 0, GN, GM);
}

#endif


#ifndef CONV_3D_W1
#define CONV_3D_W1

#ifndef CONV_3D_W1_MICRO
#define CONV_3D_W1_MICRO

#define __conv3D_W1(streams, index, length, X, IH, IW, W, Y, N, IC, OC) \
	conv3d_W1(streams, index, length, X,IH,IW, W, Y, IC,OC,\
		GET_GN(OC), GET_GM(N, IH, IW), 0, 0)

#define conv3dW1Branch(SIZE_Y, SIZE_X) {\
	int GNr = GN & (SIZE_Y), GMr = GM & (SIZE_X);\
	if(GNr && GMr){\
		int next_oc_index = (GN - GNr) + oc_index;\
		int next_j_index = (GM - GMr) + j_index;\
		conv3d_W1(streams, index, length, X,IH,IW, W, Y, IC,OC, GNr, GM, next_oc_index, j_index);\
		conv3d_W1(streams, index, length, X,IH,IW, W, Y, IC,OC, GN, GMr, oc_index, next_j_index);\
		conv3d_W1(streams, index, length, X,IH,IW, W, Y, IC,OC, GNr, GMr, next_oc_index, next_j_index);}\
	else if(GNr){\
		int next_oc_index = (GN - GNr) + oc_index;\
		conv3d_W1(streams, index, length, X,IH,IW, W, Y, IC,OC, GNr, GM, next_oc_index, j_index);}\
	else if(GMr){\
		int next_j_index = (GM - GMr) + j_index;\
		conv3d_W1(streams, index, length, X,IH,IW, W, Y, IC,OC, GN, GMr, oc_index, next_j_index);}}

#endif

//(1) GM = N * OH * OW;       GM >= 4, GM % 4 == 0
//(2) GN = OC;                GN >= 4, GN % 4 == 0
//(3) GK = FH * FW * IC = IC; GK >= 4, GK % 4 ==0
void conv3d_W1(jlong *streams, int &index, int length,
	const float* X, int IH, int IW,
	const float* W,//FH = FW = 1
		  float* Y,
	int IC, int OC,
	int GN, int GM,
	int oc_index, int j_index)
{
	next_cudaStream(stream, streams, index, length);

	if ((GN > 127) && (GM > 127) && !(IC & 7)) {//[128, 128], GK % 8 == 0
		conv3d_k88_W1(stream, 4, oc_index, j_index, X, IH, IW, W, Y, IC, OC, GN, GM);
		conv3dW1Branch(127, 127); return;
	}
	if ((GN > 63) && (GM > 63)) {//[64, 64]
		conv3d_k88_W1(stream, 3, oc_index, j_index, X, IH, IW, W, Y, IC, OC, GN, GM);
		conv3dW1Branch(63, 63); return;
	}

	if ((GN > 31) && (GM > 63)) {//[32, 64]
		conv3d_k48_W1(stream, 3, oc_index, j_index, X, IH, IW, W, Y, IC, OC, GN, GM);
		conv3dW1Branch(31, 63); return;
	}
	if ((GN > 63) && (GM > 31)) {//[64, 32]
		conv3d_k84_W1(stream, 3, oc_index, j_index, X, IH, IW, W, Y, IC, OC, GN, GM);
		conv3dW1Branch(63, 31);
	}

	if ((GN > 31) && (GM > 31)) {//[32, 32]
		conv3d_k44_W1(stream, 3, oc_index, j_index, X, IH, IW, W, Y, IC, OC, GN, GM);
		conv3dW1Branch(31, 31); return;
	}

	if ((GN > 15) && (GM > 63)) {//[16, 64]
		conv3d_k28_W1(stream, 3, oc_index, j_index, X, IH, IW, W, Y, IC, OC, GN, GM);
		conv3dW1Branch(15, 63); return;
	}
	if ((GN > 63) && (GM > 15)) {//[64, 16]
		conv3d_k82_W1(stream, 3, oc_index, j_index, X, IH, IW, W, Y, IC, OC, GN, GM);
		conv3dW1Branch(63, 15); return;
	}

	if ((GN > 15) && (GM > 31)) {//[16, 32]
		conv3d_k24_W1(stream, 3, oc_index, j_index, X, IH, IW, W, Y, IC, OC, GN, GM);
		conv3dW1Branch(15, 31); return;
	}
	if ((GN > 31) && (GM > 15)) {//[32, 16]
		conv3d_k42_W1(stream, 3, oc_index, j_index, X, IH, IW, W, Y, IC, OC, GN, GM);
		conv3dW1Branch(31, 15); return;
	}

	//=======[Small]===========================================
	if ((GN > 15) && (GM > 15)) {//[16, 16]
		conv3d_k22_W1(stream, 3, oc_index, j_index, X, IH, IW, W, Y, IC, OC, GN, GM);
		conv3dW1Branch(15, 15); return;
	}

	if ((GN > 15) && (GM > 7)) {//[16, 8]
		conv3d_k21_W1(stream, 3, oc_index, j_index, X, IH, IW, W, Y, IC, OC, GN, GM);
		conv3dW1Branch(15, 7); return;
	}

	if ((GN > 7) && (GM > 31)) {//[8, 32]
		conv3d_k14_W1(stream, 3, oc_index, j_index, X, IH, IW, W, Y, IC, OC, GN, GM);
		conv3dW1Branch(7, 31); return;
	}
	if ((GN > 7) && (GM > 15)) {//[8, 16]
		conv3d_k12_W1(stream, 3, oc_index, j_index, X, IH, IW, W, Y, IC, OC, GN, GM);
		conv3dW1Branch(7, 15); return;
	}
	if ((GN > 7) && (GM > 7)) {//[8, 8]
		conv3d_k11_W1(stream, 3, oc_index, j_index, X, IH, IW, W, Y, IC, OC, GN, GM);
		conv3dW1Branch(7, 7); return;
	}

	if (GN > 7) {//[8, 4]
		conv3d_k21_W1(stream, 2, oc_index, j_index, X, IH, IW, W, Y, IC, OC, GN, GM);
		conv3dW1Branch(7, 3); return;
	}

	if (GM > 15) {//[4, 16]
		conv3d_k14_W1(stream, 2, oc_index, j_index, X, IH, IW, W, Y, IC, OC, GN, GM);
		conv3dW1Branch(3, 15); return;
	}
	if (GM > 7) {//[4, 8]
		conv3d_k12_W1(stream, 2, oc_index, j_index, X, IH, IW, W, Y, IC, OC, GN, GM);
		conv3dW1Branch(3, 7); return;
	}
	conv3d_k11_W1(stream, 2, oc_index, j_index, X, IH, IW, W, Y, IC, OC, GN, GM);
}

#endif

#endif



//Implicit-GEMM: Remode the kernel: W[OC, FH, FW, IC] -> CW[FH, FW, IC, OC]
#ifndef GEMM_REMODE_AREA
#define GEMM_REMODE_AREA

#ifndef CONV_3D_GEMMR
#define CONV_3D_GEMMR

#ifndef CONV_3D_GEMMR_MICRO
#define CONV_3D_GEMMR_MICRO

#define __conv3D_GemmR(streams, index, length, X, IH, IW, W, CW, FH, FW, Y, OH, OW, N, IC, OC, sh, sw, ph, pw) \
	conv3dGemmR(streams, index, length, X,IH,IW, W,CW,FH,FW, Y,OH,OW, N,IC,OC, sh,sw,ph,pw,\
		GET_GN(OC), GET_GM(N, OH, OW), GET_GK(FH, FW, IC), 0, 0)

#define conv3dGemmRBranch(SIZE_Y, SIZE_X) {\
	int GNr = GN & (SIZE_Y), GMr = GM & (SIZE_X);\
	if(GNr && GMr){\
		int next_oc_index = (GN - GNr) + oc_index;\
		int next_j_index = (GM - GMr) + j_index;\
		conv3dGemmR(streams, index, length, X,IH,IW, W,CW,FH,FW, Y,OH,OW, N,IC,OC, sh,sw,ph,pw,\
			GNr, GM, GK, next_oc_index, j_index);\
		conv3dGemmR(streams, index, length, X,IH,IW, W,CW,FH,FW, Y,OH,OW, N,IC,OC, sh,sw,ph,pw,\
            GN, GMr, GK, oc_index, next_j_index);\
		conv3dGemmR(streams, index, length, X,IH,IW, W,CW,FH,FW, Y,OH,OW, N,IC,OC, sh,sw,ph,pw,\
            GNr, GMr, GK, next_oc_index, next_j_index);}\
	else if(GNr){\
		int next_oc_index = (GN - GNr) + oc_index;\
		conv3dGemmR(streams, index, length, X,IH,IW, W,CW,FH,FW, Y,OH,OW, N,IC,OC, sh,sw,ph,pw,\
			 GNr, GM, GK, next_oc_index, j_index);}\
	else if(GMr){\
		int next_j_index = (GM - GMr) + j_index;\
		conv3dGemmR(streams, index, length, X,IH,IW, W,CW,FH,FW, Y,OH,OW, N,IC,OC, sh,sw,ph,pw,\
			 GN, GMr, GK, oc_index, next_j_index);}}

#endif

//(1) GM = N * OH * OW;  GM >= 4, GM % 4 == 0
//(2) GN = OC;           GN >= 4, GN % 4 == 0
//(3) GK = FH * FW * IC; GK >= 8, GK % 4 == 0
//As: FH * FW >= 2, So: GK >= 8
//V2 <=> V1: j_index = ohw_index*N + n_index
void conv3dGemmR(jlong* streams, int &index, int length,
	const float* X, int IH, int IW,
	const float* W, const float* CW, int FH, int FW,
	      float* Y, int OH, int OW,
	int N, int IC, int OC,
	int sh, int sw, int ph, int pw,
	int GN, int GM, int GK,
	int oc_index, int j_index)
{
	next_cudaStream(stream, streams, index, length);

	if ((FH == 3) && (FW == 3) && (sh == 1) && (sw == 1) && (ph == 1) && (pw == 1) && (GN > 63) && (GM > 127)) {//Im2col-Winograd: [64, 128]
		bool flag = conv3D_Winograd_f2x3_k48R_p1<4>(stream, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, IC, OC, GN, GM);
		if (flag) { conv3dGemmRBranch(63, 127); return; }
	}

	//==========================================[Main Part: [128, 128], GK % 8 == 0]=========================================================================
	if ((GN > 127) && (GM > 127) && !(GK & 7)) {
		if((sh == 1) && (sw == 1) && conv3D_Gemm_uernel_s1_ruse_8x8R<4>(stream, oc_index, j_index, X, IH, IW, CW, FH, FW, Y, OH, OW, IC, OC, ph, pw, GN, GM));
		else if (conv3D_Gemm_uernel_8x8R<4>(stream, oc_index, j_index, X, IH, IW, CW, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM));
		//======[FW, IC are power of 2]======================================================================================================================
		else if (IS_POWER2(FW) && IS_POWER2(IC)) {
			if (!(OH & 3) && !(OW & 3)) conv3dGemm_k88R4_fw_ic2pow(stream, 4, oc_index, j_index, X, IH, IW, CW, FH, LOG2(FW), Y, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);
			else if (!(N & 7) && (j_index == 0) && !(GM & 127)) conv3dGemm_k88RA_fw_ic2pow(stream, 4, oc_index, j_index, X, IH, IW, CW, FH, LOG2(FW), Y, OH, OW, N, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);
			else conv3dGemm_k88R_fw_ic2pow(stream, 4, oc_index, j_index, X, IH, IW, CW, FH, LOG2(FW), Y, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);
		}
		//======[FH = FW = 3]================================================================================================================================
		else if ((FH == 3) && (FW == 3)) {
			if (!(OH & 3) && !(OW & 3)) {
				if (IS_POWER2(IC)) conv3dGemm_k88R4W3_ic2pow(stream, 4, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);
				else conv3dGemm_k88R4W3(stream, 4, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
			}
			else if (!(N & 7) && (j_index == 0) && !(GM & 127)) {
				if (IS_POWER2(IC)) conv3dGemm_k88RA_W3_ic2pow(stream, 4, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, N, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);
				else conv3dGemm_k88RA_W3(stream, 4, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, N, IC, OC, sh, sw, ph, pw, GN, GM);
			}
			else if (IS_POWER2(IC)) conv3dGemm_k88RW3_ic2pow(stream, 4, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);
			else conv3dGemm_k88RW3(stream, 4, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
		}
		//======[FH = FW = 5]================================================================================================================================
		else if ((FH == 5) && (FW == 5)) {
			if (!(OH & 3) && !(OW & 3)) {
				if (IS_POWER2(IC)) conv3dGemm_k88R4W5_ic2pow(stream, 4, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);
				else conv3dGemm_k88R4W5(stream, 4, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
			}
			else if (!(N & 7) && (j_index == 0) && !(GM & 127)) {//Kernel As
				if (IS_POWER2(IC)) conv3dGemm_k88RA_W5_ic2pow(stream, 4, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, N, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);
				else conv3dGemm_k88RA_W5(stream, 4, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, N, IC, OC, sh, sw, ph, pw, GN, GM);
			}
			else if (IS_POWER2(IC)) conv3dGemm_k88RW5_ic2pow(stream, 4, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);
			else conv3dGemm_k88RW5(stream, 4, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
		}
		//======[Common]=====================================================================================================================================
		else if (!(OH & 3) && !(OW & 3)) {
			if (IS_POWER2(IC)) conv3dGemm_k88R4_ic2pow(stream, 4, oc_index, j_index, X, IH, IW, CW, FH, FW, Y, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);
			else conv3dGemm_k88R4(stream, 4, oc_index, j_index, X, IH, IW, CW, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
		}
		else if (!(N & 7) && (j_index == 0) && !(GM & 127)) {
			if (IS_POWER2(IC)) conv3dGemm_k88RA_ic2pow(stream, 4, oc_index, j_index, X, IH, IW, CW, FH, FW, Y, OH, OW, N, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);
			else conv3dGemm_k88RA(stream, 4, oc_index, j_index, X, IH, IW, CW, FH, FW, Y, OH, OW, N, IC, OC, sh, sw, ph, pw, GN, GM);
		}
		else if (IS_POWER2(IC)) conv3dGemm_k88R_ic2pow(stream, 4, oc_index, j_index, X, IH, IW, CW, FH, FW, Y, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);
		else conv3dGemm_k88R(stream, 4, oc_index, j_index, X, IH, IW, CW, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
		conv3dGemmRBranch(127, 127); return;
	}

	//==========================================[Main Part: [64, 64], GK % 4 == 0]===========================================================================
	if ((GN > 63) && (GM > 63)) {//[64, 64]
		if ((sh == 1) && (sw == 1) && conv3D_Gemm_uernel_s1_ruse_8x8R<3>(stream, oc_index, j_index, X, IH, IW, CW, FH, FW, Y, OH, OW, IC, OC, ph, pw, GN, GM));
		else if (conv3D_Gemm_uernel_8x8R<3>(stream, oc_index, j_index, X, IH, IW, CW, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM));
		//======[FW, IC are power of 2]======================================================================================================================
		else if (IS_POWER2(FW) && IS_POWER2(IC)) {
			if (!(OH & 3) && !(OW & 3)) conv3dGemm_k88R4_fw_ic2pow(stream, 3, oc_index, j_index, X, IH, IW, CW, FH, LOG2(FW), Y, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);
			else if (!(N & 7) && (j_index == 0) && !(GM & 63)) conv3dGemm_k88RA_fw_ic2pow(stream, 3, oc_index, j_index, X, IH, IW, CW, FH, LOG2(FW), Y, OH, OW, N, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);
			else conv3dGemm_k88R_fw_ic2pow(stream, 3, oc_index, j_index, X, IH, IW, CW, FH, LOG2(FW), Y, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);
		}
		//======[FH = FW = 3]================================================================================================================================
		else if ((FH == 3) && (FW == 3)) {
			if (!(OH & 3) && !(OW & 3)) {
				if (IS_POWER2(IC)) conv3dGemm_k88R4W3_ic2pow(stream, 3, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);
				else conv3dGemm_k88R4W3(stream, 3, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
			}
			else if (!(N & 7) && (j_index == 0) && !(GM & 63)) {//Kernel A
				if (IS_POWER2(IC)) conv3dGemm_k88RA_W3_ic2pow(stream, 3, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, N, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);
				else conv3dGemm_k88RA_W3(stream, 3, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, N, IC, OC, sh, sw, ph, pw, GN, GM);
			}
			else if (IS_POWER2(IC)) conv3dGemm_k88RW3_ic2pow(stream, 3, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);
			else conv3dGemm_k88RW3(stream, 3, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
		}
		//======[FH = FW = 5]================================================================================================================================
		else if ((FH == 5) && (FW == 5)) {
			if (!(OH & 3) && !(OW & 3)) {
				if (IS_POWER2(IC)) conv3dGemm_k88R4W5_ic2pow(stream, 3, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);
				else conv3dGemm_k88R4W5(stream, 3, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
			}
			else if (!(N & 7) && (j_index == 0) && !(GM & 63)) {
				if (IS_POWER2(IC)) conv3dGemm_k88RA_W5_ic2pow(stream, 3, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, N, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);
				else conv3dGemm_k88RA_W5(stream, 3, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, N, IC, OC, sh, sw, ph, pw, GN, GM);
			}
			else if (IS_POWER2(IC)) conv3dGemm_k88RW5_ic2pow(stream, 3, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);
			else conv3dGemm_k88RW5(stream, 3, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
		}
		//======[Common]=====================================================================================================================================
		else if (!(OH & 3) && !(OW & 3)) {
			if (IS_POWER2(IC)) conv3dGemm_k88R4_ic2pow(stream, 3, oc_index, j_index, X, IH, IW, CW, FH, FW, Y, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);
			else conv3dGemm_k88R4(stream, 3, oc_index, j_index, X, IH, IW, CW, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
		}
		else if (!(N & 7) && (j_index == 0) && !(GM & 63)) {
			if (IS_POWER2(IC)) conv3dGemm_k88RA_ic2pow(stream, 3, oc_index, j_index, X, IH, IW, CW, FH, FW, Y, OH, OW, N, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);
			else conv3dGemm_k88RA(stream, 3, oc_index, j_index, X, IH, IW, CW, FH, FW, Y, OH, OW, N, IC, OC, sh, sw, ph, pw, GN, GM);
		}
		else if (IS_POWER2(IC)) conv3dGemm_k88R_ic2pow(stream, 3, oc_index, j_index, X, IH, IW, CW, FH, FW, Y, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);
		else conv3dGemm_k88R(stream, 3, oc_index, j_index, X, IH, IW, CW, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
		conv3dGemmRBranch(63, 63); return;
	}

	//==========================================[Main Part: [32-64, 32-64], GK % 4 == 0]=====================================================================
	if ((GN > 63) && (GM > 31)) {//[64, 32]
		if(!(IC & 7)) conv3dPure_u84R(stream, 3, oc_index, j_index, X, IH, IW, CW, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
		else if (IS_POWER2(FW) && IS_POWER2(IC)) conv3dGemm_k84R_fw_ic2pow(stream, 3, oc_index, j_index, X, IH, IW, CW, FH, LOG2(FW), Y, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);
		else conv3dPure_k84R(stream, 3, oc_index, j_index, X, IH, IW, CW, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
		conv3dGemmRBranch(63, 31); return;
	}
	if ((GN > 31) && (GM > 63)) {//[32, 64]
		if ((FH == 3) && (FW == 3) && (sh == 1) && (sw == 1) && (ph == 1) && (pw == 1) && conv3D_Winograd_f2x3_k48R_p1<3>(stream, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, IC, OC, GN, GM));
		else if (!(IC & 7)) conv3dPure_u48R(stream, 3, oc_index, j_index, X, IH, IW, CW, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
		else conv3dPure_k48R(stream, 3, oc_index, j_index, X, IH, IW, CW, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
		conv3dGemmRBranch(31, 63); return;
	}

	if ((GN > 31) && (GM > 31)) {//[32, 32]
		if (!(IC & 7)) conv3dPure_u44R(stream, 3, oc_index, j_index, X, IH, IW, CW, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
		else conv3dPure_k44R(stream, 3, oc_index, j_index, X, IH, IW, CW, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
		conv3dGemmRBranch(31, 31); return;
	}

	if ((GN > 63) && (GM > 15)) {//[64, 16]
		conv3dPure_k82R(stream, 3, oc_index, j_index, X, IH, IW, CW, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
		conv3dGemmRBranch(63, 15); return;
	}
	if ((GN > 15) && (GM > 63)) {//[16, 64]
		conv3dPure_k28(stream, 3, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
		conv3dGemmRBranch(15, 63); return;
	}

	if ((GN > 31) && (GM > 15)) {//[32, 16]
		conv3dPure_k42R(stream, 3, oc_index, j_index, X, IH, IW, CW, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
		conv3dGemmBranch(31, 15); return;
	}
	if ((GN > 15) && (GM > 31)) {//[16, 32]
		conv3dPure_k24(stream, 3, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
		conv3dGemmBranch(15, 31); return;
	}

	//==========================================[Small]======================================================================================================
	if ((GN > 15) && (GM > 15)) {//[16, 16], GK >= 8
		conv3dGemm_k22(stream, 3, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
		conv3dGemmBranch(15, 15); return;
	}

	if ((GN > 31) && (GM > 7)) {//[32, 8], GK >= 8
		conv3dGemm_k41(stream, 3, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
		conv3dGemmBranch(31, 7); return;
	}
	if ((GN > 15) && (GM > 7)) {//[16, 8], GK >= 8
		conv3dGemm_k21(stream, 3, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
		conv3dGemmBranch(15, 7); return;
	}

	if ((GN > 7) && (GM > 31)) {//[8, 32], GK >= 8
		conv3dGemm_k14(stream, 3, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
		conv3dGemmBranch(7, 31); return;
	}
	if ((GN > 7) && (GM > 15)) {//[8, 16], GK >= 8
		conv3dGemm_k12(stream, 3, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
		conv3dGemmBranch(7, 15); return;
	}
	if ((GN > 7) && (GM > 7)) {//[8, 8], GK >= 8
		conv3dGemm_k11(stream, 3, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
		conv3dGemmBranch(7, 7); return;
	}
	
	if (GN > 15) {//[16, 4]
		conv3dGemm_k41(stream, 2, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
		conv3dGemmBranch(15, 3); return;
	}
	if (GN > 7) {//[8, 4]
		conv3dGemm_k21(stream, 2, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
		conv3dGemmBranch(7, 3); return;
	}

	if (GM > 31) {//[4, 32], GK >= 8
		conv3dGemm_s1_4x2(stream, 3, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
		conv3dGemmBranch(3, 31); return;
	}
	if (GM > 15) {//[4, 16], GK >= 8
		conv3dGemm_s1_2x2(stream, 3, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
		conv3dGemmBranch(3, 15); return;
	}
	if (GM > 7) {//[4, 8]
		conv3dGemm_k12(stream, 2, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
		conv3dGemmBranch(3, 7); return;
	}
	conv3dGemm_k11(stream, 2, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
}

#endif


#ifndef CONV_3D_GEMMR_TEXTURE
#define CONV_3D_GEMMR_TEXTURE

#ifndef CONV_3D_GEMMR_TEXTURE_MICRO
#define CONV_3D_GEMMR_TEXTURE_MICRO

#define __conv3D_GemmR_tex(streams, index, length, texX, X, IH, IW, W, CW, FH, FW, Y, OH, OW, N, IC, OC, sh, sw, ph, pw) \
	conv3dGemmR_texture(streams, index, length, texX,X,IH,IW, W,CW,FH,FW, Y,OH,OW, N,IC,OC, sh,sw,ph,pw,\
		GET_GN(OC), GET_GM(N, OH, OW), GET_GK(FH, FW, IC), 0, 0)

#define conv3dGemmRBranch_tex(SIZE_Y, SIZE_X) {\
	int GNr = GN & (SIZE_Y), GMr = GM & (SIZE_X);\
	if(GNr && GMr){\
		int next_oc_index = (GN - GNr) + oc_index;\
		int next_j_index = (GM - GMr) + j_index;\
		conv3dGemmR_texture(streams, index, length, texX,X,IH,IW, W,CW,FH,FW, Y,OH,OW, N,IC,OC, sh,sw,ph,pw,\
			GNr, GM, GK, next_oc_index, j_index);\
		conv3dGemmR_texture(streams, index, length, texX,X,IH,IW, W,CW,FH,FW, Y,OH,OW, N,IC,OC, sh,sw,ph,pw,\
            GN, GMr, GK, oc_index, next_j_index);\
		conv3dGemmR_texture(streams, index, length, texX,X,IH,IW, W,CW,FH,FW, Y,OH,OW, N,IC,OC, sh,sw,ph,pw,\
            GNr, GMr, GK, next_oc_index, next_j_index);}\
	else if(GNr){\
		int next_oc_index = (GN - GNr) + oc_index;\
		conv3dGemmR_texture(streams, index, length, texX,X,IH,IW, W,CW,FH,FW, Y,OH,OW, N,IC,OC, sh,sw,ph,pw,\
			 GNr, GM, GK, next_oc_index, j_index);}\
	else if(GMr){\
		int next_j_index = (GM - GMr) + j_index;\
		conv3dGemmR_texture(streams, index, length, texX,X,IH,IW, W,CW,FH,FW, Y,OH,OW, N,IC,OC, sh,sw,ph,pw,\
			 GN, GMr, GK, oc_index, next_j_index);}}

#endif

//(1) GM = N * OH * OW;  GM >= 4, GM % 4 == 0
//(2) GN = OC;           GN >= 4, GN % 4 == 0
//(3) GK = FH * FW * IC; GK >= 8, GK % 4 == 0
//As: FH * FW >= 2, so: GK >= 8
//V2 <=> V1: j_index = ohw_index*N + n_index
void conv3dGemmR_texture(jlong* streams, int &index, int length,
	cudaTextureObject_t texX, const float* X, int IH, int IW,
	const float*           W, const float* CW, int FH, int FW,
	      float*           Y, int OH, int OW,
	int N, int IC, int OC,
	int sh, int sw, int ph, int pw,
	int GN, int GM, int GK,
	int oc_index, int j_index)
{
	next_cudaStream(stream, streams, index, length);

	if ((FH == 3) && (FW == 3) && (sh == 1) && (sw == 1) && !(OW & 1) && !(IC & 7) && (GN > 63) && (GM > 127) ) {//Im2col-Winograd: [64, 128], OW % 2 == 0
		if (!(OW & 3)) conv3dWinograd_f2x3_k64x128R4_tex(stream, oc_index, j_index, texX, IH, IW, CW, 3, Y, OH, OW, IC, OC, ph, pw, GN, GM);
		else conv3dWinograd_f2x3_k64x128R_tex(stream, oc_index, j_index, texX, IH, IW, CW, 3, Y, OH, OW, IC, OC, ph, pw, GN, GM);
		conv3dGemmRBranch_tex(63, 127); return;
	}

	//==========================================[Main Part: [128, 128], GK % 8 == 0]=========================================================================
	if ((GN > 127) && (GM > 127) && !(GK & 7)) {
		if ((sh == 1) && (sw == 1) && conv3D_Gemm_uernel_s1_ruse_8x8R<4>(stream, oc_index, j_index, X, IH, IW, CW, FH, FW, Y, OH, OW, IC, OC, ph, pw, GN, GM));
		else if (conv3D_Gemm_uernel_8x8R<4>(stream, oc_index, j_index, X, IH, IW, CW, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM));
		//======[FW, IC are power of 2]======================================================================================================================
		else if (IS_POWER2(FW) && IS_POWER2(IC)) {
			if (!(OH & 3) && !(OW & 3)) conv3dGemm_k88R4_fw_ic2pow_tex(stream, 4, oc_index, j_index, texX, IH, IW, CW, FH, LOG2(FW), Y, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);
			else if (!(N & 7) && (j_index == 0) && !(GM & 127)) conv3dGemm_k88RA_fw_ic2pow_tex(stream, 4, oc_index, j_index, texX, IH, IW, CW, FH, LOG2(FW), Y, OH, OW, N, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);
			else conv3dGemm_k88R_fw_ic2pow_tex(stream, 4, oc_index, j_index, texX, IH, IW, CW, FH, LOG2(FW), Y, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);
		}
		//======[FH = FW = 3]================================================================================================================================
		else if ((FH == 3) && (FW == 3)) {
			if (!(OH & 3) && !(OW & 3)) {
				if (IS_POWER2(IC)) conv3dGemm_k88R4W3_ic2pow(stream, 4, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);
				else conv3dGemm_k88R4W3(stream, 4, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
			}
			else if (!(N & 7) && (j_index == 0) && !(GM & 127)) {
				if (IS_POWER2(IC)) conv3dGemm_k88RA_W3_ic2pow(stream, 4, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, N, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);
				else conv3dGemm_k88RA_W3_tex(stream, 4, oc_index, j_index, texX, IH, IW, CW, Y, OH, OW, N, IC, OC, sh, sw, ph, pw, GN, GM);
			}
			else if (IS_POWER2(IC)) conv3dGemm_k88RW3_ic2pow(stream, 4, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);
			else conv3dGemm_k88RW3(stream, 4, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
		}
		//======[FH = FW = 5]================================================================================================================================
		else if ((FH == 5) && (FW == 5)) {
			if (!(OH & 3) && !(OW & 3)) {
				if (IS_POWER2(IC)) conv3dGemm_k88R4W5_ic2pow(stream, 4, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);
				else conv3dGemm_k88R4W5(stream, 4, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
			}
			else if (!(N & 7) && (j_index == 0) && !(GM & 127)) {//Kernel As
				if (IS_POWER2(IC)) conv3dGemm_k88RA_W5_ic2pow(stream, 4, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, N, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);
				else conv3dGemm_k88RA_W5_tex(stream, 4, oc_index, j_index, texX, IH, IW, CW, Y, OH, OW, N, IC, OC, sh, sw, ph, pw, GN, GM);
			}
			else if (IS_POWER2(IC)) conv3dGemm_k88RW5_ic2pow(stream, 4, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);
			else conv3dGemm_k88RW5(stream, 4, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
		}
		//======[Common]=====================================================================================================================================
		else if (!(OH & 3) && !(OW & 3)) {
			if (IS_POWER2(IC)) {
				if ((FH == 7) && (FW == 7)) conv3dGemm_k88R4W7_ic2pow_tex(stream, 4, oc_index, j_index, texX, IH, IW, CW, Y, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);
				else conv3dGemm_k88R4_ic2pow_tex(stream, 4, oc_index, j_index, texX, IH, IW, CW, FH, FW, Y, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);
			}
			else conv3dGemm_k88R4_tex(stream, 4, oc_index, j_index, texX, IH, IW, CW, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
		}
		else if (!(N & 7) && (j_index == 0) && !(GM & 127)) {
			if (IS_POWER2(IC)) conv3dGemm_k88RA_ic2pow(stream, 4, oc_index, j_index, X, IH, IW, CW, FH, FW, Y, OH, OW, N, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);
			else conv3dGemm_k88RA(stream, 4, oc_index, j_index, X, IH, IW, CW, FH, FW, Y, OH, OW, N, IC, OC, sh, sw, ph, pw, GN, GM);
		}
		else if (IS_POWER2(IC)) conv3dGemm_k88R_ic2pow(stream, 4, oc_index, j_index, X, IH, IW, CW, FH, FW, Y, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);
		else conv3dGemm_k88R(stream, 4, oc_index, j_index, X, IH, IW, CW, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
		conv3dGemmRBranch_tex(127, 127); return;
	}

	//==========================================[Main Part: [32-64, 32-64], GK % 4 == 0]=====================================================================
	if ((GN > 63) && (GM > 63)) {
		if ((sh == 1) && (sw == 1) && conv3D_Gemm_uernel_s1_ruse_8x8R<3>(stream, oc_index, j_index, X, IH, IW, CW, FH, FW, Y, OH, OW, IC, OC, ph, pw, GN, GM));
		else if (conv3D_Gemm_uernel_8x8R<3>(stream, oc_index, j_index, X, IH, IW, CW, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM));
		//======[FW, IC are power of 2]======================================================================================================================
		else if (IS_POWER2(FW) && IS_POWER2(IC)) {
			if (!(OH & 3) && !(OW & 3)) conv3dGemm_k88R4_fw_ic2pow_tex(stream, 3, oc_index, j_index, texX, IH, IW, CW, FH, LOG2(FW), Y, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);
			else if (!(N & 7) && (j_index == 0) && !(GM & 63)) conv3dGemm_k88RA_fw_ic2pow_tex(stream, 3, oc_index, j_index, texX, IH, IW, CW, FH, LOG2(FW), Y, OH, OW, N, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);
			else conv3dGemm_k88R_fw_ic2pow_tex(stream, 3, oc_index, j_index, texX, IH, IW, CW, FH, LOG2(FW), Y, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);
		}
		//======[FH = FW = 3]================================================================================================================================
		else if ((FH == 3) && (FW == 3)) {
			if (!(OH & 3) && !(OW & 3)) {
				if (IS_POWER2(IC)) conv3dGemm_k88R4W3_ic2pow_tex(stream, 3, oc_index, j_index, texX, IH, IW, CW, Y, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);
				else conv3dGemm_k88R4W3_tex(stream, 3, oc_index, j_index, texX, IH, IW, CW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
			}
			else if (!(N & 7) && (j_index == 0) && !(GM & 63)) {//Kernel A
				if (IS_POWER2(IC)) conv3dGemm_k88RA_W3_ic2pow_tex(stream, 3, oc_index, j_index, texX, IH, IW, CW, Y, OH, OW, N, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);
				else conv3dGemm_k88RA_W3_tex(stream, 3, oc_index, j_index, texX, IH, IW, CW, Y, OH, OW, N, IC, OC, sh, sw, ph, pw, GN, GM);
			}
			else if (IS_POWER2(IC)) conv3dGemm_k88RW3_ic2pow_tex(stream, 3, oc_index, j_index, texX, IH, IW, CW, Y, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);
			else conv3dGemm_k88RW3_tex(stream, 3, oc_index, j_index, texX, IH, IW, CW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
		}
		//======[FH = FW = 5]================================================================================================================================
		else if ((FH == 5) && (FW == 5)) {
			if (!(OH & 3) && !(OW & 3)) {
				if (IS_POWER2(IC)) conv3dGemm_k88R4W5_ic2pow_tex(stream, 3, oc_index, j_index, texX, IH, IW, CW, Y, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);
				else conv3dGemm_k88R4W5_tex(stream, 3, oc_index, j_index, texX, IH, IW, CW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
			}
			else if (!(N & 7) && (j_index == 0) && !(GM & 63)) {
				if (IS_POWER2(IC))  conv3dGemm_k88RA_W5_ic2pow_tex(stream, 3, oc_index, j_index, texX, IH, IW, CW, Y, OH, OW, N, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);
				else conv3dGemm_k88RA_W5_tex(stream, 3, oc_index, j_index, texX, IH, IW, CW, Y, OH, OW, N, IC, OC, sh, sw, ph, pw, GN, GM);
			}
			else if (IS_POWER2(IC)) conv3dGemm_k88RW5_ic2pow_tex(stream, 3, oc_index, j_index, texX, IH, IW, CW, Y, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);
			else conv3dGemm_k88RW5_tex(stream, 3, oc_index, j_index, texX, IH, IW, CW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
		}
		//======[Common]=====================================================================================================================================
		else if (!(OH & 3) && !(OW & 3)) {
			if (IS_POWER2(IC)) {
				if ((FH == 7) && (FW == 7)) conv3dGemm_k88R4W7_ic2pow_tex(stream, 3, oc_index, j_index, texX, IH, IW, CW, Y, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);
				else conv3dGemm_k88R4_ic2pow_tex(stream, 3, oc_index, j_index, texX, IH, IW, CW, FH, FW, Y, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);
			}
			else conv3dGemm_k88R4_tex(stream, 3, oc_index, j_index, texX, IH, IW, CW, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
		}
		else if (!(N & 7) && (j_index == 0) && !(GM & 127)) {
			if (IS_POWER2(IC)) conv3dGemm_k88RA_ic2pow(stream, 3, oc_index, j_index, X, IH, IW, CW, FH, FW, Y, OH, OW, N, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);
			else conv3dGemm_k88RA(stream, 3, oc_index, j_index, X, IH, IW, CW, FH, FW, Y, OH, OW, N, IC, OC, sh, sw, ph, pw, GN, GM);
		}
		else if (IS_POWER2(IC)) conv3dGemm_k88R_ic2pow(stream, 3, oc_index, j_index, X, IH, IW, CW, FH, FW, Y, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);
		else conv3dGemm_k88R(stream, 3, oc_index, j_index, X, IH, IW, CW, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
		conv3dGemmRBranch_tex(63, 63); return;
	}

	//==========================================[Main Part: [32-64, 32-64], GK % 4 == 0]=====================================================================
	if ((GN > 63) && (GM > 31)) {//[64, 32]
		if ((FH == 3) && (FW == 3) && (sh == 1) && (sw == 1) && conv3D_Winograd_f2x3_k48R_tex<3>(stream, oc_index, j_index, texX, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, GN, GM));
		else if (!(IC & 7)) conv3dPure_u84R(stream, 3, oc_index, j_index, X, IH, IW, CW, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
		else if (IS_POWER2(FW) && IS_POWER2(IC)) conv3dGemm_k84R_fw_ic2pow(stream, 3, oc_index, j_index, X, IH, IW, CW, FH, LOG2(FW), Y, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);
		else conv3dPure_k84R(stream, 3, oc_index, j_index, X, IH, IW, CW, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
		conv3dGemmRBranch_tex(63, 31); return;
	}

	if ((GN > 31) && (GM > 63)) {//[32, 64]
		if (!(IC & 7)) conv3dPure_u48R(stream, 3, oc_index, j_index, X, IH, IW, CW, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
		else conv3dPure_k48R_tex(stream, 3, oc_index, j_index, texX, IH, IW, CW, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
		conv3dGemmRBranch(31, 63); return;
	}

	if ((GN > 31) && (GM > 31)) {//[32, 32]
		if (!(IC & 7)) conv3dPure_u44R(stream, 3, oc_index, j_index, X, IH, IW, CW, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
		else conv3dPure_k44R(stream, 3, oc_index, j_index, X, IH, IW, CW, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
		conv3dGemmRBranch(31, 31); return;
	}

	if ((GN > 63) && (GM > 15)) {//[64, 16]
		conv3dPure_k82R(stream, 3, oc_index, j_index, X, IH, IW, CW, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
		conv3dGemmRBranch(63, 15); return;
	}
	if ((GN > 15) && (GM > 63)) {//[16, 64]
		conv3dPure_k28(stream, 3, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
		conv3dGemmRBranch(15, 63); return;
	}

	if ((GN > 31) && (GM > 15)) {//[32, 16]
		conv3dPure_k42R(stream, 3, oc_index, j_index, X, IH, IW, CW, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
		conv3dGemmBranch(31, 15); return;
	}
	if ((GN > 15) && (GM > 31)) {//[16, 32]
		conv3dPure_k24(stream, 3, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
		conv3dGemmBranch(15, 31); return;
	}
		
	//==========================================[Small]======================================================================================================
	if ((GN > 15) && (GM > 15)) {//[16, 16], GK >= 8
		conv3dGemm_k22(stream, 3, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
		conv3dGemmBranch(15, 15); return;
	}

	if ((GN > 31) && (GM > 7)) {//[32, 8], GK >= 8
		conv3dGemm_k41(stream, 3, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
		conv3dGemmBranch(31, 7); return;
	}
	if ((GN > 15) && (GM > 7)) {//[16, 8], GK >= 8
		conv3dGemm_k21(stream, 3, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
		conv3dGemmBranch(15, 7); return;
	}

	if ((GN > 7) && (GM > 31)) {//[8, 32], GK >= 8
		conv3dGemm_k14(stream, 3, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
		conv3dGemmBranch(7, 31); return;
	}
	if ((GN > 7) && (GM > 15)) {//[8, 16], GK >= 8
		conv3dGemm_k12(stream, 3, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
		conv3dGemmBranch(7, 15); return;
	}
	if ((GN > 7) && (GM > 7)) {//[8, 8], GK >= 8
		conv3dGemm_k11(stream, 3, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
		conv3dGemmBranch(7, 7); return;
	}
	
	if (GN > 15) {//[16, 4]
		conv3dGemm_k41(stream, 2, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
		conv3dGemmBranch(15, 3); return;
	}
	if (GN > 7) {//[8, 4]
		conv3dGemm_k21(stream, 2, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
		conv3dGemmBranch(7, 3); return;
	}

	if (GM > 31) {//[4, 32], GK >= 8
		conv3dGemm_s1_4x2(stream, 3, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
		conv3dGemmBranch(3, 31); return;
	}
	if (GM > 15) {//[4, 16], GK >= 8
		conv3dGemm_s1_2x2(stream, 3, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
		conv3dGemmBranch(3, 15); return;
	}
	if (GM > 7) {//[4, 8]
		conv3dGemm_k12(stream, 2, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
		conv3dGemmBranch(3, 7); return;
	}
	conv3dGemm_k11(stream, 2, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
}

#endif


#ifndef CONV_3D_GEMMV2R
#define CONV_3D_GEMMV2R

#ifndef CONV_3D_GEMMV2R_MICRO
#define CONV_3D_GEMMV2R_MICRO

#define __conv3D_GemmV2R(env, streams, index, length, X, IH, IW, sizeX, W, CW, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw) \
	conv3dGemmV2R(env, streams,index,length, X,IH,IW,sizeX, W,CW,FH,FW, Y,OH,OW, IC,OC, sh,sw,ph,pw,\
		GET_GN(OC), N, 0, 0)

#define conv3dGemmV2RBranch(SIZE_Y, SIZE_X) {\
	int GNr = GN & (SIZE_Y), Nr = N & (SIZE_X);\
	if(GNr && Nr){\
		int next_oc_index = (GN - GNr) + oc_index;\
		int next_n_index = (N - Nr) + n_index;\
		conv3dGemmV2R(env,streams,index,length, X,IH,IW,sizeX, W,CW,FH,FW, Y,OH,OW, IC,OC, sh,sw,ph,pw,\
			GNr, N, next_oc_index, n_index);\
		conv3dGemmV2R(env,streams,index,length, X,IH,IW,sizeX, W,CW,FH,FW, Y,OH,OW, IC,OC, sh,sw,ph,pw,\
            GN, Nr, oc_index, next_n_index);\
		conv3dGemmV2R(env,streams,index,length, X,IH,IW,sizeX, W,CW,FH,FW, Y,OH,OW, IC,OC, sh,sw,ph,pw,\
            GNr, Nr, next_oc_index, next_n_index);}\
	else if(GNr){\
		int next_oc_index = (GN - GNr) + oc_index;\
		conv3dGemmV2R(env,streams,index,length, X,IH,IW,sizeX, W,CW,FH,FW, Y,OH,OW, IC,OC, sh,sw,ph,pw,\
			 GNr, N, next_oc_index, n_index);}\
	else if(Nr){\
		int next_n_index = (N - Nr) + n_index;\
		conv3dGemmV2R(env,streams,index,length, X,IH,IW,sizeX, W,CW,FH,FW, Y,OH,OW, IC,OC, sh,sw,ph,pw,\
			 GN, Nr, oc_index, next_n_index);}}

#endif

//(1) GM = N * OH * OW;  GM >= 4, GM % 4 == 0
//(2) GN = OC;           GN >= 4, GN % 4 == 0
//(3) GK = FH * FW * IC; GK >= 8, GK % 4 == 0
//when: N >= 64, OC >= 64, V2 will be enabled
void conv3dGemmV2R(JNIEnv *env, jlong* streams, int &index, int length,
	const float* X, int IH, int IW, int sizeX,
	const float* W, const float* CW, int FH, int FW,
	      float* Y, int OH, int OW,
	int IC, int OC,//(GN, GM, bz) = (OC, N, OH*OW)
	int sh, int sw, int ph, int pw,
	int GN, int N,//GM = N, N != X.N 
	int oc_index, int n_index)//j_index = n_index * OH_OW
{
	next_cudaStream(stream, streams, index, length);

	//======[Stage1]=============================================================================================================================
	if ((GN > 127) && (N > 127) && !(IC & 7)) {//[128, 128], IC % 8 == 0
		if (IS_POWER2(IC)) {
			if (CAN_V2_W3P1) {
				if (!(IC & 15)) conv3dGemmV2_u88RW3p1_ic2pow(stream, 4, oc_index, n_index, X, IH, IW, CW, Y, OH, OW, LOG2(IC), OC, sh, sw, GN, N);
				else conv3dGemmV2_k88RW3p1_ic2pow(stream, 4, oc_index, n_index, X, IH, IW, CW, Y, OH, OW, LOG2(IC), OC, sh, sw, GN, N);
			}
			else if (CAN_V2_W4P1) {
				if (!(IC & 15)) conv3dGemmV2_u88RW4p1_ic2pow(stream, 4, oc_index, n_index, X, IH, IW, CW, Y, OH, OW, LOG2(IC), OC, sh, sw, GN, N);
				else conv3dGemmV2_k88RW4p1_ic2pow(stream, 4, oc_index, n_index, X, IH, IW, CW, Y, OH, OW, LOG2(IC), OC, sh, sw, GN, N);
			}
			else if (CAN_V2_W5P2) {
				if (!(IC & 15)) conv3dGemmV2_u88RW5p2_ic2pow(stream, 4, oc_index, n_index, X, IH, IW, CW, Y, OH, OW, LOG2(IC), OC, sh, sw, GN, N);
				else conv3dGemmV2_k88RW5p2_ic2pow(stream, 4, oc_index, n_index, X, IH, IW, CW, Y, OH, OW, LOG2(IC), OC, sh, sw, GN, N);
			}
			else if (!(IC & 15)) conv3dGemmV2_u88R(stream, 4, oc_index, n_index, X, IH, IW, CW, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, N);
			else conv3dGemmV2_k88R(stream, 4, oc_index, n_index, X, IH, IW, CW, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, N);
		}
		else if (!(IC & 15)) conv3dGemmV2_u88R(stream, 4, oc_index, n_index, X, IH, IW, CW, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, N);
		else conv3dGemmV2_k88R(stream, 4, oc_index, n_index, X, IH, IW, CW, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, N);
		conv3dGemmV2RBranch(127, 127); return;
	}

	//======[Stage2]=============================================================================================================================
	float scale_up = PADDING_SCALE_UP(IH, IW, OH, OW, FH, FW, sh, sw);
	if (scale_up < 1.3f) {//gate1: to conv3dGemmR
		V2_TO_V1(FH, FW, OH, OW, N, IC, n_index);
		if (index > 0) index = index - 1;

		if (env) {//env != null, use texture
			cudaTextureObject_t texX = floatTexture((float*)X, sizeX, env);
			conv3dGemmR_texture(streams, index, length,
				texX, X, IH, IW, 
				W, CW, FH, FW, 
				Y, OH, OW,
				N, IC, OC,
				sh, sw, ph, pw,
				GN, GM, GK, 
				oc_index, j_index);
			cudaError_t error = cudaDestroyTextureObject(texX); handleError(error);
			return;
		}

		printf("1231231\n");
		conv3dGemmR(streams, index, length,
			X, IH, IW, 
			W, CW, FH, FW, 
			Y, OH, OW, 
			N, IC, OC,
			sh, sw, ph, pw,
			GN, GM, GK,
			oc_index, j_index);
		return;
	}

	if ((GN > 63) && (N > 63)) {//[64, 64], IC % 4 == 0
		if (IS_POWER2(IC)) {
			if (CAN_V2_W3P1) {
				if (!(IC & 7)) conv3dGemmV2_u88RW3p1_ic2pow(stream, 3, oc_index, n_index, X, IH, IW, CW, Y, OH, OW, LOG2(IC), OC, sh, sw, GN, N);
				else conv3dGemmV2_k88RW3p1_ic2pow(stream, 3, oc_index, n_index, X, IH, IW, CW, Y, OH, OW, LOG2(IC), OC, sh, sw, GN, N);
			}
			else if (CAN_V2_W4P1) {
				if (!(IC & 7)) conv3dGemmV2_u88RW4p1_ic2pow(stream, 3, oc_index, n_index, X, IH, IW, CW, Y, OH, OW, LOG2(IC), OC, sh, sw, GN, N);
				else conv3dGemmV2_k88RW4p1_ic2pow(stream, 3, oc_index, n_index, X, IH, IW, CW, Y, OH, OW, LOG2(IC), OC, sh, sw, GN, N);
			}
			else if (CAN_V2_W5P2) {
				if (!(IC & 7)) conv3dGemmV2_u88RW5p2_ic2pow(stream, 3, oc_index, n_index, X, IH, IW, CW, Y, OH, OW, LOG2(IC), OC, sh, sw, GN, N);
				else conv3dGemmV2_k88RW5p2_ic2pow(stream, 3, oc_index, n_index, X, IH, IW, CW, Y, OH, OW, LOG2(IC), OC, sh, sw, GN, N);
			}
			else if (!(IC & 7)) conv3dGemmV2_u88R(stream, 3, oc_index, n_index, X, IH, IW, CW, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, N);
			else conv3dGemmV2_k88R(stream, 3, oc_index, n_index, X, IH, IW, CW, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, N);
		}
		else if (!(IC & 7)) conv3dGemmV2_u88R(stream, 3, oc_index, n_index, X, IH, IW, CW, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, N);
		else conv3dGemmV2_k88R(stream, 3, oc_index, n_index, X, IH, IW, CW, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, N);
		conv3dGemmV2RBranch(63, 63); return;
	}

	//======[Stage3]=============================================================================================================================
	if (scale_up < 1.4f) {//gate2: to conv3dGemmR
		V2_TO_V1(FH, FW, OH, OW, N, IC, n_index);
		if (index > 0) index = index - 1;

		if (env) {//env != null, use texture
			cudaTextureObject_t texX = floatTexture((float*)X, sizeX, env);
			conv3dGemmR_texture(streams, index, length,
				texX, X, IH, IW, W, CW, FH, FW, Y, OH, OW, N, IC, OC, sh, sw, ph, pw, 
				GN, GM, GK, oc_index, j_index);
			cudaError_t error = cudaDestroyTextureObject(texX); handleError(error);
			return;
		}

		conv3dGemmR(streams, index, length,
			X, IH, IW, W, CW, FH, FW, Y, OH, OW, N, IC, OC, sh, sw, ph, pw,
			GN, GM, GK, oc_index, j_index);
		return;
	}

	if ((GN > 63) && (N > 31)) {
		conv3dGemmV2_k84R(stream, 3, oc_index, n_index, X, IH, IW, CW, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, N);
		conv3dGemmV2RBranch(63, 31); return;
	}
	if ((GN > 31) && (N > 63)) {
		conv3dGemmV2_k48R(stream, 3, oc_index, n_index, X, IH, IW, CW, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, N);
		conv3dGemmV2RBranch(31, 63); return;
	}

	if ((GN > 31) && (N > 31)) {
		conv3dGemmV2_k44R(stream, 3, oc_index, n_index, X, IH, IW, CW, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, N);
		conv3dGemmV2RBranch(31, 31); return;
	}

	if ((GN > 31) && (N > 15)) {
		conv3dGemmV2_k42R(stream, 3, oc_index, n_index, X, IH, IW, CW, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, N);
		conv3dGemmV2RBranch(31, 15); return;
	}
	if ((GN > 15) && (N > 31)) {
		conv3dGemmV2_k24R(stream, 3, oc_index, n_index, X, IH, IW, CW, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, N);
		conv3dGemmV2RBranch(15, 31); return;
	}

	V2_TO_V1(FH, FW, OH, OW, N, IC, n_index);
	if (index > 0) index = index - 1;

	if (env) {//gate3: env != null, use texture
		cudaTextureObject_t texX = floatTexture((float*)X, sizeX, env);
		conv3dGemmR_texture(streams, index, length,
			texX, X, IH, IW, W, CW, FH, FW, Y, OH, OW, N, IC, OC, sh, sw, ph, pw,
			GN, GM, GK, oc_index, j_index);
		cudaError_t error = cudaDestroyTextureObject(texX); handleError(error);
		return;
	}

	conv3dGemmR(streams, index, length,
		X, IH, IW, W, CW, FH, FW, Y, OH, OW, N, IC, OC, sh, sw, ph, pw,
		GN, GM, GK, oc_index, j_index);
}

#endif


#ifndef CONV_3D_GEMMR_W1
#define CONV_3D_GEMMR_W1

#ifndef CONV_3D_GEMMR_W1_MICRO
#define CONV_3D_GEMMR_W1_MICRO

#define __conv3D_GemmR_W1(streams, index, length, X, IH, IW, W, CW, Y, N, IC, OC) \
	conv3d_GemmR_W1(streams, index, length, X,IH,IW, W,CW, Y, IC,OC,\
		GET_GN(OC), GET_GM(N, IH, IW), 0, 0)

#define conv3d_GemmR_W1Branch(SIZE_Y, SIZE_X) {\
	int GNr = GN & (SIZE_Y), GMr = GM & (SIZE_X);\
	if(GNr && GMr){\
		int next_oc_index = (GN - GNr) + oc_index;\
		int next_j_index = (GM - GMr) + j_index;\
		conv3d_GemmR_W1(streams, index, length, X,IH,IW, W,CW, Y, IC,OC, GNr, GM, next_oc_index, j_index);\
		conv3d_GemmR_W1(streams, index, length, X,IH,IW, W,CW, Y, IC,OC, GN, GMr, oc_index, next_j_index);\
		conv3d_GemmR_W1(streams, index, length, X,IH,IW, W,CW, Y, IC,OC, GNr, GMr, next_oc_index, next_j_index);}\
	else if(GNr){\
		int next_oc_index = (GN - GNr) + oc_index;\
		conv3d_GemmR_W1(streams, index, length, X,IH,IW, W,CW, Y, IC,OC, GNr, GM, next_oc_index, j_index);}\
	else if(GMr){\
		int next_j_index = (GM - GMr) + j_index;\
		conv3d_GemmR_W1(streams, index, length, X,IH,IW, W,CW, Y, IC,OC, GN, GMr, oc_index, next_j_index);}}

#endif

//(1) GM = N * OH * OW;       GM >= 4, GM % 4 == 0
//(2) GN = OC;                GN >= 4, GN % 4 == 0
//(3) GK = FH * FW * IC = IC; GK >= 4, GK % 4 == 0
void conv3d_GemmR_W1(jlong *streams, int &index, int length,
	const float* X, int IH, int IW,
	const float* W, const float* CW,//FH = FW = 1
	      float* Y,
	int IC, int OC,
	int GN, int GM,
	int oc_index, int j_index)
{
	next_cudaStream(stream, streams, index, length);

	if ((GN > 127) && (GM > 127) && !(IC & 7)) {//[128, 128], GK % 8 == 0, remode
		if (!(IC & 15)) conv3d_u88R_W1(stream, 4, oc_index, j_index, X, IH, IW, CW, Y, IC, OC, GN, GM);
		else conv3d_k88R_W1(stream, 4, oc_index, j_index, X, IH, IW, CW, Y, IC, OC, GN, GM);
		conv3d_GemmR_W1Branch(127, 127); return;
	}
	if ((GN > 63) && (GM > 63)) {//[64, 64], remode
		if (!(IC & 7)) conv3d_u88R_W1(stream, 3, oc_index, j_index, X, IH, IW, CW, Y, IC, OC, GN, GM);
		else conv3d_k88R_W1(stream, 3, oc_index, j_index, X, IH, IW, CW, Y, IC, OC, GN, GM);
		conv3d_GemmR_W1Branch(63, 63); return;
	}

	if ((GN > 31) && (GM > 63)) {//[32, 64], remode
		conv3d_k48R_W1(stream, 3, oc_index, j_index, X, IH, IW, CW, Y, IC, OC, GN, GM);
		conv3d_GemmR_W1Branch(31, 63); return;
	}
	if ((GN > 63) && (GM > 31)) {//[64, 32], remode
		conv3d_k84R_W1(stream, 3, oc_index, j_index, X, IH, IW, CW, Y, IC, OC, GN, GM);
		conv3d_GemmR_W1Branch(63, 31);
	}

	if ((GN > 31) && (GM > 31)) {//[32, 32], remode
		conv3d_k44R_W1(stream, 3, oc_index, j_index, X, IH, IW, CW, Y, IC, OC, GN, GM);
		conv3d_GemmR_W1Branch(31, 31); return;
	}

	if ((GN > 15) && (GM > 63)) {//[16, 64]
		conv3d_k28_W1(stream, 3, oc_index, j_index, X, IH, IW, W, Y, IC, OC, GN, GM);
		conv3d_GemmR_W1Branch(15, 63); return;
	}
	if ((GN > 63) && (GM > 15)) {//[64, 16], remode
		conv3d_k82R_W1(stream, 3, oc_index, j_index, X, IH, IW, CW, Y, IC, OC, GN, GM);
		conv3d_GemmR_W1Branch(63, 15); return;
	}

	if ((GN > 15) && (GM > 31)) {//[16, 32]
		conv3d_k24_W1(stream, 3, oc_index, j_index, X, IH, IW, W, Y, IC, OC, GN, GM);
		conv3dW1Branch(15, 31); return;
	}
	if ((GN > 31) && (GM > 15)) {//[32, 16], remode
		conv3d_k42R_W1(stream, 3, oc_index, j_index, X, IH, IW, CW, Y, IC, OC, GN, GM);
		conv3dW1Branch(31, 15); return;
	}

	//=======[Small]================================================
	if ((GN > 15) && (GM > 15)) {//[16, 16]
		conv3d_k22_W1(stream, 3, oc_index, j_index, X, IH, IW, W, Y, IC, OC, GN, GM);
		conv3dW1Branch(15, 15); return;
	}
	
	if ((GN > 15) && (GM > 7)) {//[16, 8]
		conv3d_k21_W1(stream, 3, oc_index, j_index, X, IH, IW, W, Y, IC, OC, GN, GM);
		conv3dW1Branch(15, 7); return;
	}

	if ((GN > 7) && (GM > 31)) {//[8, 32]
		conv3d_k14_W1(stream, 3, oc_index, j_index, X, IH, IW, W, Y, IC, OC, GN, GM);
		conv3dW1Branch(7, 31); return;
	}
	if ((GN > 7) && (GM > 15)) {//[8, 16]
		conv3d_k12_W1(stream, 3, oc_index, j_index, X, IH, IW, W, Y, IC, OC, GN, GM);
		conv3dW1Branch(7, 15); return;
	}
	if ((GN > 7) && (GM > 7)) {//[8, 8]
		conv3d_k11_W1(stream, 3, oc_index, j_index, X, IH, IW, W, Y, IC, OC, GN, GM);
		conv3dW1Branch(7, 7); return;
	}

	if (GN > 7) {//[8, 4]
		conv3d_k21_W1(stream, 2, oc_index, j_index, X, IH, IW, W, Y, IC, OC, GN, GM);
		conv3dW1Branch(7, 3); return;
	}

	if (GM > 15) {//[4, 16]
		conv3d_k14_W1(stream, 2, oc_index, j_index, X, IH, IW, W, Y, IC, OC, GN, GM);
		conv3dW1Branch(3, 15); return;
	}
	if (GM > 7) {
		conv3d_k12_W1(stream, 2, oc_index, j_index, X, IH, IW, W, Y, IC, OC, GN, GM);
		conv3dW1Branch(3, 7); return;
	}
	conv3d_k11_W1(stream, 2, oc_index, j_index, X, IH, IW, W, Y, IC, OC, GN, GM);
}

#endif

#endif



//Im2col-Winograd: Remode the kernel: W[OC, FH, FW, IC] -> CW[FH, FW, IC, OC]
#ifndef IM2COL_WINOGRAD_REMODE_AREA
#define IM2COL_WINOGRAD_REMODE_AREA

#ifndef CONV_3D_IM2COL_WINOGRADR_MICRO
#define CONV_3D_IM2COL_WINOGRADR_MICRO

#define im2col_winograd_GemmR_Gate(GNr, GMr) {\
	if(GNr && GMr){\
		int next_oc_index = (GN - GNr);\
		int next_j_index  = (GM - GMr);\
		conv3dGemmR(streams, index, length, X,IH,IW, W,CW,FH,FW, Y,OH,OW, N,IC,OC, 1,1,ph,pw,\
			GNr, GM, GK, next_oc_index, 0);\
		conv3dGemmR(streams, index, length, X,IH,IW, W,CW,FH,FW, Y,OH,OW, N,IC,OC, 1,1,ph,pw,\
            GN, GMr, GK, 0, next_j_index);\
		conv3dGemmR(streams, index, length, X,IH,IW, W,CW,FH,FW, Y,OH,OW, N,IC,OC, 1,1,ph,pw,\
            GNr, GMr, GK, next_oc_index, next_j_index);}\
	else if(GNr){\
		int next_oc_index = (GN - GNr);\
		conv3dGemmR(streams, index, length, X,IH,IW, W,CW,FH,FW, Y,OH,OW, N,IC,OC, 1,1,ph,pw,\
			 GNr, GM, GK, next_oc_index, 0);}\
	else if(GMr){\
		int next_j_index = (GM - GMr);\
		conv3dGemmR(streams, index, length, X,IH,IW, W,CW,FH,FW, Y,OH,OW, N,IC,OC, 1,1,ph,pw,\
			 GN, GMr, GK, 0, next_j_index);}}

#define im2col_winograd_GemmR_Gate_tex(GNr, GMr) {\
	if(GNr && GMr){\
		int next_oc_index = (GN - GNr);\
		int next_j_index  = (GM - GMr);\
		conv3dGemmR_texture(streams, index, length, texX,X,IH,IW, W,CW,FH,FW, Y,OH,OW, N,IC,OC, 1,1,ph,pw,\
			GNr, GM, GK, next_oc_index, 0);\
		conv3dGemmR_texture(streams, index, length, texX,X,IH,IW, W,CW,FH,FW, Y,OH,OW, N,IC,OC, 1,1,ph,pw,\
            GN, GMr, GK, 0, next_j_index);\
		conv3dGemmR_texture(streams, index, length, texX,X,IH,IW, W,CW,FH,FW, Y,OH,OW, N,IC,OC, 1,1,ph,pw,\
            GNr, GMr, GK, next_oc_index, next_j_index);}\
	else if(GNr){\
		int next_oc_index = (GN - GNr);\
		conv3dGemmR_texture(streams, index, length, texX,X,IH,IW, W,CW,FH,FW, Y,OH,OW, N,IC,OC, 1,1,ph,pw,\
			 GNr, GM, GK, next_oc_index, 0);}\
	else if(GMr){\
		int next_j_index = (GM - GMr);\
		conv3dGemmR_texture(streams, index, length, texX,X,IH,IW, W,CW,FH,FW, Y,OH,OW, N,IC,OC, 1,1,ph,pw,\
			 GN, GMr, GK, 0, next_j_index);}}

#endif


#ifndef CONV_3D_IM2COL_WINOGRADR_S8_R
#define CONV_3D_IM2COL_WINOGRADR_S8_R

#define __conv3D_Im2col_Winograd_s8_tex(env, streams, index, length, X, texX, IH, IW, W, CW, FH, FW, Y, OH, OW, N, IC, OC, ph, pw)\
	conv3d_Im2col_Winograd_s8_texture(env, streams, index, length, X,texX,IH,IW, W,CW,FH,FW, Y,OH,OW, N,IC,OC, ph,pw,\
		GET_GN(OC), GET_GM(N, OH, OW))

//(1) IC % 8 == 0
//(2) env != NULL: useTexture
//(3) GM = N * OH * OW;  GM >= 4, GM % 4 == 0
//(4) GN = OC;           GN >= 4, GN % 4 == 0
//(5) (FH, FW) belongs to [2, 3, 4, 5, 6, 7]
//(6) GN: OC   >= 64
//(7) GM: N*OH >= 32
bool conv3d_Im2col_Winograd_s8_texture(JNIEnv *env, jlong* streams, int &index, int length,
	const float* X, cudaTextureObject_t texX, int IH, int IW,
	const float* W, const float* CW, int FH, int FW,
	      float* Y, int OH, int OW,
	int N, int IC, int OC,
	int ph, int pw,//sh = sw = 1
	int GN, int GM)
{
	int GNr, GMr;

	//==============[FW = 2x + 1]=======================================================================================================================================
	if ((FW == 3) && (OW >= 6)) {//F(6, 3)
		if (FH == 3) { conv3D_Winograd_s8_W3_64x32R_tex<3>(streams, index, length, X, texX, IH, IW, CW, Y, OH, OW, N, IC, OC, ph, pw, GN, GM, GNr, GMr); goto END_TRUE; }//3*3
		if (FH == 5) { conv3D_Winograd_s8_W3_64x32R_tex<5>(streams, index, length, X, texX, IH, IW, CW, Y, OH, OW, N, IC, OC, ph, pw, GN, GM, GNr, GMr); goto END_TRUE; }
		if (FH == 7) { conv3D_Winograd_s8_W3_64x32R_tex<7>(streams, index, length, X, texX, IH, IW, CW, Y, OH, OW, N, IC, OC, ph, pw, GN, GM, GNr, GMr); goto END_TRUE; }
		if (FH == 4) { conv3D_Winograd_s8_W3_64x32R_tex<4>(streams, index, length, X, texX, IH, IW, CW, Y, OH, OW, N, IC, OC, ph, pw, GN, GM, GNr, GMr); goto END_TRUE; }
		if (FH == 6) { conv3D_Winograd_s8_W3_64x32R_tex<6>(streams, index, length, X, texX, IH, IW, CW, Y, OH, OW, N, IC, OC, ph, pw, GN, GM, GNr, GMr); goto END_TRUE; }
		if (FH == 8) { conv3D_Winograd_s8_W3_64x32R_tex<8>(streams, index, length, X, texX, IH, IW, CW, Y, OH, OW, N, IC, OC, ph, pw, GN, GM, GNr, GMr); goto END_TRUE; }
		goto END_FALSE;
	}

	if ((FW == 5) && (OW >= 4)) {//F(4, 5)
		if (FH == 5) { conv3D_Winograd_s8_W5_64x32R_tex<5>(streams, index, length, X, texX, IH, IW, CW, Y, OH, OW, N, IC, OC, ph, pw, GN, GM, GNr, GMr); goto END_TRUE; }//5*5
		if (FH == 3) { conv3D_Winograd_s8_W5_64x32R_tex<3>(streams, index, length, X, texX, IH, IW, CW, Y, OH, OW, N, IC, OC, ph, pw, GN, GM, GNr, GMr); goto END_TRUE; }
		if (FH == 7) { conv3D_Winograd_s8_W5_64x32R_tex<7>(streams, index, length, X, texX, IH, IW, CW, Y, OH, OW, N, IC, OC, ph, pw, GN, GM, GNr, GMr); goto END_TRUE; }
		if (FH == 4) { conv3D_Winograd_s8_W5_64x32R_tex<4>(streams, index, length, X, texX, IH, IW, CW, Y, OH, OW, N, IC, OC, ph, pw, GN, GM, GNr, GMr); goto END_TRUE; }
		if (FH == 6) { conv3D_Winograd_s8_W5_64x32R_tex<6>(streams, index, length, X, texX, IH, IW, CW, Y, OH, OW, N, IC, OC, ph, pw, GN, GM, GNr, GMr); goto END_TRUE; }
		if (FH == 8) { conv3D_Winograd_s8_W5_64x32R_tex<8>(streams, index, length, X, texX, IH, IW, CW, Y, OH, OW, N, IC, OC, ph, pw, GN, GM, GNr, GMr); goto END_TRUE; }
		goto END_FALSE;
	}

	if ((FW == 7) && (OW >= 2)) {//F(2, 7)
		if (FH == 7) { conv3D_Winograd_s8_W7_64x32R_tex<7>(streams, index, length, X, texX, IH, IW, CW, Y, OH, OW, N, IC, OC, ph, pw, GN, GM, GNr, GMr); goto END_TRUE; }//7*7
		if (FH == 3) { conv3D_Winograd_s8_W7_64x32R_tex<3>(streams, index, length, X, texX, IH, IW, CW, Y, OH, OW, N, IC, OC, ph, pw, GN, GM, GNr, GMr); goto END_TRUE; }
		if (FH == 5) { conv3D_Winograd_s8_W7_64x32R_tex<5>(streams, index, length, X, texX, IH, IW, CW, Y, OH, OW, N, IC, OC, ph, pw, GN, GM, GNr, GMr); goto END_TRUE; }
		if (FH == 4) { conv3D_Winograd_s8_W7_64x32R_tex<4>(streams, index, length, X, texX, IH, IW, CW, Y, OH, OW, N, IC, OC, ph, pw, GN, GM, GNr, GMr); goto END_TRUE; }
		if (FH == 6) { conv3D_Winograd_s8_W7_64x32R_tex<6>(streams, index, length, X, texX, IH, IW, CW, Y, OH, OW, N, IC, OC, ph, pw, GN, GM, GNr, GMr); goto END_TRUE; }
		if (FH == 8) { conv3D_Winograd_s8_W7_64x32R_tex<8>(streams, index, length, X, texX, IH, IW, CW, Y, OH, OW, N, IC, OC, ph, pw, GN, GM, GNr, GMr); goto END_TRUE; }
		goto END_FALSE;
	}

	//==============[FW = 2x]===========================================================================================================================================
	if ((FW == 2) && (OW >= 7)) {//F(7, 2)
		if (FH == 2) { conv3D_Winograd_s8_W2_64x32R_tex<2>(streams, index, length, X, texX, IH, IW, CW, Y, OH, OW, N, IC, OC, ph, pw, GN, GM, GNr, GMr); goto END_TRUE; }//2*2
		if (FH == 4) { conv3D_Winograd_s8_W2_64x32R_tex<4>(streams, index, length, X, texX, IH, IW, CW, Y, OH, OW, N, IC, OC, ph, pw, GN, GM, GNr, GMr); goto END_TRUE; }
		if (FH == 6) { conv3D_Winograd_s8_W2_64x32R_tex<6>(streams, index, length, X, texX, IH, IW, CW, Y, OH, OW, N, IC, OC, ph, pw, GN, GM, GNr, GMr); goto END_TRUE; }
		if (FH == 3) { conv3D_Winograd_s8_W2_64x32R_tex<3>(streams, index, length, X, texX, IH, IW, CW, Y, OH, OW, N, IC, OC, ph, pw, GN, GM, GNr, GMr); goto END_TRUE; }
		if (FH == 5) { conv3D_Winograd_s8_W2_64x32R_tex<5>(streams, index, length, X, texX, IH, IW, CW, Y, OH, OW, N, IC, OC, ph, pw, GN, GM, GNr, GMr); goto END_TRUE; }
		if (FH == 7) { conv3D_Winograd_s8_W2_64x32R_tex<7>(streams, index, length, X, texX, IH, IW, CW, Y, OH, OW, N, IC, OC, ph, pw, GN, GM, GNr, GMr); goto END_TRUE; }
		goto END_FALSE;
	}

	if ((FW == 4) && (OW >= 5)) {//F(5, 4)
		if (FH == 4) { conv3D_Winograd_s8_W4_64x32R_tex<4>(streams, index, length, X, texX, IH, IW, CW, Y, OH, OW, N, IC, OC, ph, pw, GN, GM, GNr, GMr); goto END_TRUE; }//4*4
		if (FH == 2) { conv3D_Winograd_s8_W4_64x32R_tex<2>(streams, index, length, X, texX, IH, IW, CW, Y, OH, OW, N, IC, OC, ph, pw, GN, GM, GNr, GMr); goto END_TRUE; }
		if (FH == 6) { conv3D_Winograd_s8_W4_64x32R_tex<6>(streams, index, length, X, texX, IH, IW, CW, Y, OH, OW, N, IC, OC, ph, pw, GN, GM, GNr, GMr); goto END_TRUE; }
		if (FH == 3) { conv3D_Winograd_s8_W4_64x32R_tex<3>(streams, index, length, X, texX, IH, IW, CW, Y, OH, OW, N, IC, OC, ph, pw, GN, GM, GNr, GMr); goto END_TRUE; }
		if (FH == 5) { conv3D_Winograd_s8_W4_64x32R_tex<5>(streams, index, length, X, texX, IH, IW, CW, Y, OH, OW, N, IC, OC, ph, pw, GN, GM, GNr, GMr); goto END_TRUE; }
		if (FH == 7) { conv3D_Winograd_s8_W4_64x32R_tex<7>(streams, index, length, X, texX, IH, IW, CW, Y, OH, OW, N, IC, OC, ph, pw, GN, GM, GNr, GMr); goto END_TRUE; }
		goto END_FALSE;
	}

	if ((FW == 6) && (OW >= 3)) {//F[6, 3]
		if (FH == 6) { conv3D_Winograd_s8_W6_64x32R_tex<6>(streams, index, length, X, texX, IH, IW, CW, Y, OH, OW, N, IC, OC, ph, pw, GN, GM, GNr, GMr); goto END_TRUE; }//6*6
		if (FH == 2) { conv3D_Winograd_s8_W6_64x32R_tex<2>(streams, index, length, X, texX, IH, IW, CW, Y, OH, OW, N, IC, OC, ph, pw, GN, GM, GNr, GMr); goto END_TRUE; }
		if (FH == 4) { conv3D_Winograd_s8_W6_64x32R_tex<4>(streams, index, length, X, texX, IH, IW, CW, Y, OH, OW, N, IC, OC, ph, pw, GN, GM, GNr, GMr); goto END_TRUE; }
		if (FH == 3) { conv3D_Winograd_s8_W6_64x32R_tex<3>(streams, index, length, X, texX, IH, IW, CW, Y, OH, OW, N, IC, OC, ph, pw, GN, GM, GNr, GMr); goto END_TRUE; }
		if (FH == 5) { conv3D_Winograd_s8_W6_64x32R_tex<5>(streams, index, length, X, texX, IH, IW, CW, Y, OH, OW, N, IC, OC, ph, pw, GN, GM, GNr, GMr); goto END_TRUE; }
		if (FH == 7) { conv3D_Winograd_s8_W6_64x32R_tex<7>(streams, index, length, X, texX, IH, IW, CW, Y, OH, OW, N, IC, OC, ph, pw, GN, GM, GNr, GMr); goto END_TRUE; }
		goto END_FALSE;
	}

END_FALSE:
	return false;

END_TRUE:
	const int GK = FH * FW * IC; 
	if (index > 0) index--;//save L2 cache: texX != X
	if (env) { im2col_winograd_GemmR_Gate_tex(GNr, GMr); }
	else { im2col_winograd_GemmR_Gate(GNr, GMr); }
	return true;
}

#endif


#ifndef CONV_3D_IM2COL_WINOGRADR_S16_R
#define CONV_3D_IM2COL_WINOGRADR_S16_R

#define __conv3D_Im2col_Winograd_s16_tex(env, streams, index, length, X, texX, IH, IW, W, CW, FH, FW, Y, OH, OW, N, IC, OC, ph, pw)\
	conv3d_Im2col_Winograd_s16_texture(env, streams, index, length, X,texX,IH,IW, W,CW,FH,FW, Y,OH,OW, N,IC,OC, ph,pw,\
		GET_GN(OC), GET_GM(N, OH, OW))

//(1) IC % 8 == 0
//(2) env != NULL: useTexture
//(3) GM = N * OH * OW;  GM >= 4, GM % 4 == 0
//(4) GN = OC;           GN >= 4, GN % 4 == 0
//(5) (FH, FW) belongs to [7, 9]
//(6) GN: OC   >= 32
//(7) GM: N*OH >= 32
bool conv3d_Im2col_Winograd_s16_texture(JNIEnv *env, jlong* streams, int &index, int length,
	const float* X, cudaTextureObject_t texX, int IH, int IW,
	const float* W, const float* CW, int FH, int FW,
	      float* Y, int OH, int OW,
	int N, int IC, int OC,
	int ph, int pw,//sh = sw = 1
	int GN, int GM)
{
	int GNr, GMr;

	//(pw <= (FW/2)) && (OW - IW - pw + (FW - 1)/2 <= 0)
	//7: 7/2 = 3, 6/2 = 3
	//9: 9/2 = 4, 8/2 = 4
	//8: 8/2 = 4, 7/2 = 3

	//==============[FW = 2x + 1]=======================================================================================================================================
	if ((FW == 7) && (OW >= 10) && (pw <= 3) && (OW - IW - pw + 3 <= 0)) {//F(10, 7), pw <= 3
		if (FH == 7) { if (!(OC & 63)) conv3D_Winograd_s16_W7_64x32R_p3_tex<7>(streams, index, length, X, texX, IH, IW, CW, Y, OH, OW, N, IC, OC, ph, pw, GN, GM, GNr, GMr);
			           else            conv3D_Winograd_s16_W7_32x32R_p3_tex<7>(streams, index, length, X, texX, IH, IW, CW, Y, OH, OW, N, IC, OC, ph, pw, GN, GM, GNr, GMr); goto END_TRUE; }//7*7
		if (FH == 3) { if (!(OC & 63)) conv3D_Winograd_s16_W7_64x32R_p3_tex<3>(streams, index, length, X, texX, IH, IW, CW, Y, OH, OW, N, IC, OC, ph, pw, GN, GM, GNr, GMr);
			           else            conv3D_Winograd_s16_W7_32x32R_p3_tex<3>(streams, index, length, X, texX, IH, IW, CW, Y, OH, OW, N, IC, OC, ph, pw, GN, GM, GNr, GMr); goto END_TRUE; }
		if (FH == 5) { if (!(OC & 63)) conv3D_Winograd_s16_W7_64x32R_p3_tex<5>(streams, index, length, X, texX, IH, IW, CW, Y, OH, OW, N, IC, OC, ph, pw, GN, GM, GNr, GMr);
			           else            conv3D_Winograd_s16_W7_32x32R_p3_tex<5>(streams, index, length, X, texX, IH, IW, CW, Y, OH, OW, N, IC, OC, ph, pw, GN, GM, GNr, GMr); goto END_TRUE; }
		if (FH == 9) { if (!(OC & 63)) conv3D_Winograd_s16_W7_64x32R_p3_tex<9>(streams, index, length, X, texX, IH, IW, CW, Y, OH, OW, N, IC, OC, ph, pw, GN, GM, GNr, GMr);
					   else            conv3D_Winograd_s16_W7_32x32R_p3_tex<9>(streams, index, length, X, texX, IH, IW, CW, Y, OH, OW, N, IC, OC, ph, pw, GN, GM, GNr, GMr); goto END_TRUE; }
		if (FH == 2) { if (!(OC & 63)) conv3D_Winograd_s16_W7_64x32R_p3_tex<2>(streams, index, length, X, texX, IH, IW, CW, Y, OH, OW, N, IC, OC, ph, pw, GN, GM, GNr, GMr);
				       else            conv3D_Winograd_s16_W7_32x32R_p3_tex<2>(streams, index, length, X, texX, IH, IW, CW, Y, OH, OW, N, IC, OC, ph, pw, GN, GM, GNr, GMr); goto END_TRUE; }
		if (FH == 4) { if (!(OC & 63)) conv3D_Winograd_s16_W7_64x32R_p3_tex<4>(streams, index, length, X, texX, IH, IW, CW, Y, OH, OW, N, IC, OC, ph, pw, GN, GM, GNr, GMr);
				       else            conv3D_Winograd_s16_W7_32x32R_p3_tex<4>(streams, index, length, X, texX, IH, IW, CW, Y, OH, OW, N, IC, OC, ph, pw, GN, GM, GNr, GMr); goto END_TRUE; }
		if (FH == 6) { if (!(OC & 63)) conv3D_Winograd_s16_W7_64x32R_p3_tex<6>(streams, index, length, X, texX, IH, IW, CW, Y, OH, OW, N, IC, OC, ph, pw, GN, GM, GNr, GMr);
				       else            conv3D_Winograd_s16_W7_32x32R_p3_tex<6>(streams, index, length, X, texX, IH, IW, CW, Y, OH, OW, N, IC, OC, ph, pw, GN, GM, GNr, GMr); goto END_TRUE; }
		if (FH == 8) { if (!(OC & 63)) conv3D_Winograd_s16_W7_64x32R_p3_tex<8>(streams, index, length, X, texX, IH, IW, CW, Y, OH, OW, N, IC, OC, ph, pw, GN, GM, GNr, GMr);
				       else            conv3D_Winograd_s16_W7_32x32R_p3_tex<8>(streams, index, length, X, texX, IH, IW, CW, Y, OH, OW, N, IC, OC, ph, pw, GN, GM, GNr, GMr); goto END_TRUE; }
		goto END_FALSE;
	}

	if ((FW == 9) && (OW >= 8) && (pw <= 4) && (OW - IW - pw + 4 <= 0)) {//F(8, 9), pw <= 4
		if (FH == 9) { if (!(OC & 63)) conv3D_Winograd_s16_W9_64x32R_p4_tex<9>(streams, index, length, X, texX, IH, IW, CW, Y, OH, OW, N, IC, OC, ph, pw, GN, GM, GNr, GMr);
			           else            conv3D_Winograd_s16_W9_32x32R_p4_tex<9>(streams, index, length, X, texX, IH, IW, CW, Y, OH, OW, N, IC, OC, ph, pw, GN, GM, GNr, GMr); goto END_TRUE; }//9*9
		if (FH == 3) { if (!(OC & 63)) conv3D_Winograd_s16_W9_64x32R_p4_tex<3>(streams, index, length, X, texX, IH, IW, CW, Y, OH, OW, N, IC, OC, ph, pw, GN, GM, GNr, GMr);
			           else            conv3D_Winograd_s16_W9_32x32R_p4_tex<3>(streams, index, length, X, texX, IH, IW, CW, Y, OH, OW, N, IC, OC, ph, pw, GN, GM, GNr, GMr); goto END_TRUE; }
		if (FH == 5) { if (!(OC & 63)) conv3D_Winograd_s16_W9_64x32R_p4_tex<5>(streams, index, length, X, texX, IH, IW, CW, Y, OH, OW, N, IC, OC, ph, pw, GN, GM, GNr, GMr);
			           else            conv3D_Winograd_s16_W9_32x32R_p4_tex<5>(streams, index, length, X, texX, IH, IW, CW, Y, OH, OW, N, IC, OC, ph, pw, GN, GM, GNr, GMr); goto END_TRUE; }
		if (FH == 7) { if (!(OC & 63)) conv3D_Winograd_s16_W9_64x32R_p4_tex<7>(streams, index, length, X, texX, IH, IW, CW, Y, OH, OW, N, IC, OC, ph, pw, GN, GM, GNr, GMr);
			           else            conv3D_Winograd_s16_W9_32x32R_p4_tex<7>(streams, index, length, X, texX, IH, IW, CW, Y, OH, OW, N, IC, OC, ph, pw, GN, GM, GNr, GMr); goto END_TRUE; }
		if (FH == 2) { if (!(OC & 63)) conv3D_Winograd_s16_W9_64x32R_p4_tex<2>(streams, index, length, X, texX, IH, IW, CW, Y, OH, OW, N, IC, OC, ph, pw, GN, GM, GNr, GMr);
			           else            conv3D_Winograd_s16_W9_32x32R_p4_tex<2>(streams, index, length, X, texX, IH, IW, CW, Y, OH, OW, N, IC, OC, ph, pw, GN, GM, GNr, GMr); goto END_TRUE; }
		if (FH == 4) { if (!(OC & 63)) conv3D_Winograd_s16_W9_64x32R_p4_tex<4>(streams, index, length, X, texX, IH, IW, CW, Y, OH, OW, N, IC, OC, ph, pw, GN, GM, GNr, GMr);
			           else            conv3D_Winograd_s16_W9_32x32R_p4_tex<4>(streams, index, length, X, texX, IH, IW, CW, Y, OH, OW, N, IC, OC, ph, pw, GN, GM, GNr, GMr); goto END_TRUE; }
		if (FH == 6) { if (!(OC & 63)) conv3D_Winograd_s16_W9_64x32R_p4_tex<6>(streams, index, length, X, texX, IH, IW, CW, Y, OH, OW, N, IC, OC, ph, pw, GN, GM, GNr, GMr);
			           else            conv3D_Winograd_s16_W9_32x32R_p4_tex<6>(streams, index, length, X, texX, IH, IW, CW, Y, OH, OW, N, IC, OC, ph, pw, GN, GM, GNr, GMr); goto END_TRUE; }
		if (FH == 8) { if (!(OC & 63)) conv3D_Winograd_s16_W9_64x32R_p4_tex<8>(streams, index, length, X, texX, IH, IW, CW, Y, OH, OW, N, IC, OC, ph, pw, GN, GM, GNr, GMr);
			           else            conv3D_Winograd_s16_W9_32x32R_p4_tex<8>(streams, index, length, X, texX, IH, IW, CW, Y, OH, OW, N, IC, OC, ph, pw, GN, GM, GNr, GMr); goto END_TRUE; }
		goto END_FALSE;
	}

	//==============[FW = 2x]===========================================================================================================================================
	if ((FW == 8) && (OW >= 9) && (pw <= 4) && (OW - IW - pw + 3 <= 0)) {//F(9, 8), pw <= 4
		if (FH == 8) { if (!(OC & 63)) conv3D_Winograd_s16_W8_64x32R_p4_tex<8>(streams, index, length, X, texX, IH, IW, CW, Y, OH, OW, N, IC, OC, ph, pw, GN, GM, GNr, GMr);
			           else            conv3D_Winograd_s16_W8_32x32R_p4_tex<8>(streams, index, length, X, texX, IH, IW, CW, Y, OH, OW, N, IC, OC, ph, pw, GN, GM, GNr, GMr); goto END_TRUE; }//8*8
		if (FH == 2) { if (!(OC & 63)) conv3D_Winograd_s16_W8_64x32R_p4_tex<2>(streams, index, length, X, texX, IH, IW, CW, Y, OH, OW, N, IC, OC, ph, pw, GN, GM, GNr, GMr);
			           else            conv3D_Winograd_s16_W8_32x32R_p4_tex<2>(streams, index, length, X, texX, IH, IW, CW, Y, OH, OW, N, IC, OC, ph, pw, GN, GM, GNr, GMr); goto END_TRUE; }
		if (FH == 4) { if (!(OC & 63)) conv3D_Winograd_s16_W8_64x32R_p4_tex<4>(streams, index, length, X, texX, IH, IW, CW, Y, OH, OW, N, IC, OC, ph, pw, GN, GM, GNr, GMr);
				       else            conv3D_Winograd_s16_W8_32x32R_p4_tex<4>(streams, index, length, X, texX, IH, IW, CW, Y, OH, OW, N, IC, OC, ph, pw, GN, GM, GNr, GMr); goto END_TRUE; }
		if (FH == 6) { if (!(OC & 63)) conv3D_Winograd_s16_W8_64x32R_p4_tex<6>(streams, index, length, X, texX, IH, IW, CW, Y, OH, OW, N, IC, OC, ph, pw, GN, GM, GNr, GMr);
				       else            conv3D_Winograd_s16_W8_32x32R_p4_tex<6>(streams, index, length, X, texX, IH, IW, CW, Y, OH, OW, N, IC, OC, ph, pw, GN, GM, GNr, GMr); goto END_TRUE; }
		if (FH == 3) { if (!(OC & 63)) conv3D_Winograd_s16_W8_64x32R_p4_tex<3>(streams, index, length, X, texX, IH, IW, CW, Y, OH, OW, N, IC, OC, ph, pw, GN, GM, GNr, GMr);
			           else            conv3D_Winograd_s16_W8_32x32R_p4_tex<3>(streams, index, length, X, texX, IH, IW, CW, Y, OH, OW, N, IC, OC, ph, pw, GN, GM, GNr, GMr); goto END_TRUE; }
		if (FH == 7) { if (!(OC & 63)) conv3D_Winograd_s16_W8_64x32R_p4_tex<7>(streams, index, length, X, texX, IH, IW, CW, Y, OH, OW, N, IC, OC, ph, pw, GN, GM, GNr, GMr);
			           else            conv3D_Winograd_s16_W8_32x32R_p4_tex<7>(streams, index, length, X, texX, IH, IW, CW, Y, OH, OW, N, IC, OC, ph, pw, GN, GM, GNr, GMr); goto END_TRUE; }
		if (FH == 5) { if (!(OC & 63)) conv3D_Winograd_s16_W8_64x32R_p4_tex<5>(streams, index, length, X, texX, IH, IW, CW, Y, OH, OW, N, IC, OC, ph, pw, GN, GM, GNr, GMr);
			           else            conv3D_Winograd_s16_W8_32x32R_p4_tex<5>(streams, index, length, X, texX, IH, IW, CW, Y, OH, OW, N, IC, OC, ph, pw, GN, GM, GNr, GMr); goto END_TRUE; }
		if (FH == 9) { if (!(OC & 63)) conv3D_Winograd_s16_W8_64x32R_p4_tex<9>(streams, index, length, X, texX, IH, IW, CW, Y, OH, OW, N, IC, OC, ph, pw, GN, GM, GNr, GMr);
			           else            conv3D_Winograd_s16_W8_32x32R_p4_tex<9>(streams, index, length, X, texX, IH, IW, CW, Y, OH, OW, N, IC, OC, ph, pw, GN, GM, GNr, GMr); goto END_TRUE; }
		goto END_FALSE;
	}

END_FALSE:
	return false;

END_TRUE:
	const int GK = FH * FW * IC; 
	if (index > 0) index--;//save L2 cache: texX != X
	if (env) { im2col_winograd_GemmR_Gate_tex(GNr, GMr); }
	else { im2col_winograd_GemmR_Gate(GNr, GMr); }
	return true;
}

#endif

#endif

#endif//complie-area>>>>------------------------------------------------------------


#endif