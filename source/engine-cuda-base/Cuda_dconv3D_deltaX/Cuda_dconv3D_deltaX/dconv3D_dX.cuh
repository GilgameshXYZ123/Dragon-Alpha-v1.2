#pragma once

#ifndef DECONV_3D_DELTAX_H
#define DECONV_3D_DELTAX_H

#include "dconv3D_dX_tensor_remode.cuh"
#include "dconv3D_dX_ZeroPadding.cuh"
#include "dconv3D_dX_KernelSplit.cuh"
#include "dconv3D_dX_CrossAdd_kernel.cuh"
#include "dconv3D_dX_Im2col_Winograd.cuh"
#include "conv3D_Winograd2D.cuh"


#ifdef COMPLIE//<<<<complie-area--------------------------------------------------

//zero_padding: Y[N, OH, OW, OC] -> Y[N, OHp, OWp, OC]
#ifndef ZERO_PADDING_AREA
#define ZERO_PADDING_AREA

//Dense Kernel: for sh = sw = 1
#ifndef DECONV_3D_DELTAX_ZERO_PADDING_S1
#define DECONV_3D_DELTAX_ZERO_PADDING_S1

#ifndef DECONV_3D_DELTAX_ZERO_PADDING_S1_MICRO
#define DECONV_3D_DELTAX_ZERO_PADDING_S1_MICRO

#define __dconv3D_deltaX_ZeroPadding_s1(streams, index, length, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, OC, ph, pw) \
	dconv3d_deltaX_ZeroPadding_s1(streams,index,length, deltaY,OH,OW, W,FH,FW, deltaX,IH,IW, N,IC,OC, ph,pw,\
		GET_GN_ZeroPadding(IC), GET_GM_ZeroPadding(N,IH,IW), GET_GK_ZeroPadding(OC,FH,FW), 0, 0)

#define dconv3d_dX_ZeroPadding_s1_Branch(SIZE_Y, SIZE_X) {\
	int GNr = GN & (SIZE_Y), GMr = GM & (SIZE_X);\
	if(GNr && GMr){\
		int next_ic_index = (GN - GNr) + ic_index;\
		int next_j_index = (GM - GMr) + j_index;\
		dconv3d_deltaX_ZeroPadding_s1(streams,index,length, deltaY,OH,OW, W,FH,FW, deltaX,IH,IW, N,IC,OC, ph,pw,\
			GNr, GM, GK, next_ic_index, j_index);\
		dconv3d_deltaX_ZeroPadding_s1(streams,index,length, deltaY,OH,OW, W,FH,FW, deltaX,IH,IW, N,IC,OC, ph,pw,\
			GN, GMr, GK, ic_index, next_j_index);\
		dconv3d_deltaX_ZeroPadding_s1(streams,index,length, deltaY,OH,OW, W,FH,FW, deltaX,IH,IW, N,IC,OC, ph,pw,\
			GNr, GMr, GK, next_ic_index, next_j_index);}\
	else if(GNr){\
		int next_ic_index = (GN - GNr) + ic_index;\
		dconv3d_deltaX_ZeroPadding_s1(streams,index,length, deltaY,OH,OW, W,FH,FW, deltaX,IH,IW, N,IC,OC, ph,pw,\
			GNr, GM, GK, next_ic_index, j_index);}\
	else if(GMr){\
		int next_j_index = (GM - GMr) + j_index;\
		dconv3d_deltaX_ZeroPadding_s1(streams,index,length, deltaY,OH,OW, W,FH,FW, deltaX,IH,IW, N,IC,OC, ph,pw,\
			GN, GMr, GK, ic_index, next_j_index);}}

#endif

//(1) FH*FW >= 2
//(2) GN = IC;           GN >= 4, GN%4 == 0
//(3) GM = N  * IH * IW; GM >= 4, GM%4 == 0
//(4) GK = OC * FH * FW; GK >= 8, GK%4 == 0
//(5) sh = 1, sw = 1
void dconv3d_deltaX_ZeroPadding_s1(jlong *streams, int &index, int length,
	const float* deltaY, int OH, int OW,
	const float*      W, int FH, int FW,
	      float* deltaX, int IH, int IW,
	int N, int IC, int OC,
	int ph, int pw,
	int GN, int GM, int GK,
	int ic_index, int j_index)
{
	next_cudaStream(stream, streams, index, length);

	//==========================================[Main Part: [128, 128], GK % 8 == 0]=========================================================================
	if ((GN > 127) && (GM > 127) && !(GK & 7)) {
		if (deconv3D_dX_ZeroPadding_s1_uernel_ruse_8x8<4>(stream, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, ph, pw, GN, GM)) ;//ruse uernels
		else if (deconv3D_dX_ZeroPadding_s1_uernel_8x8<4>(stream, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, OC, ph, pw, GN, GM));//common uernels
		else if (IS_POWER2(FH) && IS_POWER2(FW)) k88s1_W2pow(stream, 4, ic_index, j_index, deltaY, OH, OW, W, LOG2(FH), LOG2(FW), deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
		//======[FH = FW = 3]================================================================================================================================
		else if ((FH == 3) && (FW == 3)) {
			if (IS_POWER2(OC)) {
				if (!(IH & 3) && !(IW & 3)) k88s1W3x4_oc2pow(stream, 4, ic_index, j_index, deltaY, OH, OW, W, deltaX, IH, IW, IC, LOG2(OC), ph, pw, GN, GM);
				else if (!(GM & 127) && (j_index == 0) && !(N & 7) && !(OC & 7)) k88As1W3_oc2pow(stream, 4, ic_index, j_index, deltaY, OH, OW, W, deltaX, IH, IW, N, IC, LOG2(OC), ph, pw, GN, GM);
				else k88s1W3_oc2pow(stream, 4, ic_index, j_index, deltaY, OH, OW, W, deltaX, IH, IW, IC, LOG2(OC), ph, pw, GN, GM);
			}
			else if (!(GM & 127) && (j_index == 0) && !(N & 7) && !(OC & 7)) k88As1W3(stream, 4, ic_index, j_index, deltaY, OH, OW, W, deltaX, IH, IW, N, IC, OC, ph, pw, GN, GM);
			else if (!(IH & 3) && !(IW & 3)) f88s1x4(stream, 4, ic_index, j_index, deltaY, OH, OW, W, 3, 3, deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
			else f88s1(stream, 4, ic_index, j_index, deltaY, OH, OW, W, 3, 3, deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
		}
		//======[FH = FW = 5]================================================================================================================================
		else if ((FH == 5) && (FW == 5)) {
			if (IS_POWER2(OC)) {
				if (!(IH & 3) && !(IW & 3)) k88s1W5x4_oc2pow(stream, 4, ic_index, j_index, deltaY, OH, OW, W, deltaX, IH, IW, IC, LOG2(OC), ph, pw, GN, GM);
				else if (!(GM & 127) && (j_index == 0) && !(N & 7) && !(OC & 7)) k88As1W5_oc2pow(stream, 4, ic_index, j_index, deltaY, OH, OW, W, deltaX, IH, IW, N, IC, LOG2(OC), ph, pw, GN, GM);
				else k88s1W5_oc2pow(stream, 4, ic_index, j_index, deltaY, OH, OW, W, deltaX, IH, IW, IC, LOG2(OC), ph, pw, GN, GM);
			}
			else if (!(GM & 127) && (j_index == 0) && !(N & 7) && !(OC & 7)) k88As1W5(stream, 4, ic_index, j_index, deltaY, OH, OW, W, deltaX, IH, IW, N, IC, OC, ph, pw, GN, GM);
			else if (!(IH & 3) && !(IW & 3)) f88s1x4(stream, 4, ic_index, j_index, deltaY, OH, OW, W, 5, 5, deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
			else f88s1(stream, 4, ic_index, j_index, deltaY, OH, OW, W, 5, 5, deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
		}
		//======[FH = FW = 6]================================================================================================================================
		else if ((FH == 6) && (FW == 6)) {
			if (!(IH & 3) && !(IW & 3)) f88s1x4(stream, 4, ic_index, j_index, deltaY, OH, OW, W, 6, 6, deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
			else f88s1(stream, 4, ic_index, j_index, deltaY, OH, OW, W, 6, 6, deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
		}
		//======[FH = FW = 7]================================================================================================================================
		else if ((FH == 7) && (FW == 7)) {
			if (!(IH & 3) && !(IW & 3)) f88s1x4(stream, 4, ic_index, j_index, deltaY, OH, OW, W, 7, 7, deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
			else f88s1(stream, 4, ic_index, j_index, deltaY, OH, OW, W, 7, 7, deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
		}
		//======[Common]=====================================================================================================================================
		else if (!(N & 7) && !(OC & 7) && !(GM & 127) && (j_index == 0)) {//OC % 8 == 0 && N % 8 == 0
			if (IS_POWER2(OC)) k88As1_oc2pow(stream, 4, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, LOG2(OC), ph, pw, GN, GM);
			else k88As1(stream, 4, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, OC, ph, pw, GN, GM);
		}
		else k88s1(stream, 4, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
		dconv3d_dX_ZeroPadding_s1_Branch(127, 127); return;
	}

	//==========================================[Main Part: [64, 64], GK % 4 == 0]===========================================================================
	if ((GN > 63) && (GM > 63)) {//[64, 64], GK % 4 == 0
		if (deconv3D_dX_ZeroPadding_s1_uernel_ruse_8x8<3>(stream, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, ph, pw, GN, GM));//ruse uernels
		else if (deconv3D_dX_ZeroPadding_s1_uernel_8x8<3>(stream, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, OC, ph, pw, GN, GM));//common uernels
		else if (IS_POWER2(FH) && IS_POWER2(FW)) k88s1_W2pow(stream, 3, ic_index, j_index, deltaY, OH, OW, W, LOG2(FH), LOG2(FW), deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
		//======[FH = FW = 3]================================================================================================================================
		else if ((FH == 3) && (FW == 3)) {
			if (IS_POWER2(OC)) {
				if (!(IH & 3) && !(IW & 3)) k88s1W3x4_oc2pow(stream, 3, ic_index, j_index, deltaY, OH, OW, W, deltaX, IH, IW, IC, LOG2(OC), ph, pw, GN, GM);
				else if (!(GM & 63) && (j_index == 0) && !(N & 7)) k88As1W3_oc2pow(stream, 3, ic_index, j_index, deltaY, OH, OW, W, deltaX, IH, IW, N, IC, LOG2(OC), ph, pw, GN, GM);
				else k88s1W3_oc2pow(stream, 3, ic_index, j_index, deltaY, OH, OW, W, deltaX, IH, IW, IC, LOG2(OC), ph, pw, GN, GM);
			}
			else if (!(GM & 63) && (j_index == 0) && !(N & 7)) k88As1W3(stream, 3, ic_index, j_index, deltaY, OH, OW, W, deltaX, IH, IW, N, IC, OC, ph, pw, GN, GM);
			else if (!(IH & 3) && !(IW & 3)) f88s1x4(stream, 3, ic_index, j_index, deltaY, OH, OW, W, 3, 3, deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
			else f88s1(stream, 3, ic_index, j_index, deltaY, OH, OW, W, 3, 3, deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
		}
		//======[FH = FW = 5]================================================================================================================================
		else if ((FH == 5) && (FW == 5)) {
			if (IS_POWER2(OC)) {
				if (!(IH & 3) && !(IW & 3)) k88s1W5x4_oc2pow(stream, 3, ic_index, j_index, deltaY, OH, OW, W, deltaX, IH, IW, IC, LOG2(OC), ph, pw, GN, GM);
				else if (!(GM & 63) && (j_index == 0) && !(N & 7)) k88As1W5_oc2pow(stream, 3, ic_index, j_index, deltaY, OH, OW, W, deltaX, IH, IW, N, IC, LOG2(OC), ph, pw, GN, GM);
				else k88s1W5_oc2pow(stream, 3, ic_index, j_index, deltaY, OH, OW, W, deltaX, IH, IW, IC, LOG2(OC), ph, pw, GN, GM);
			}
			else if (!(GM & 63) && (j_index == 0) && !(N & 7)) k88As1W5(stream, 3, ic_index, j_index, deltaY, OH, OW, W, deltaX, IH, IW, N, IC, OC, ph, pw, GN, GM);
			else if (!(IH & 3) && !(IW & 3)) f88s1x4(stream, 3, ic_index, j_index, deltaY, OH, OW, W, 5, 5, deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
			else f88s1(stream, 3, ic_index, j_index, deltaY, OH, OW, W, 5, 5, deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
		}
		//======[FH = FW = 6]================================================================================================================================
		else if ((FH == 6) && (FW == 6)) {
			if (!(IH & 3) && !(IW & 3)) f88s1x4(stream, 3, ic_index, j_index, deltaY, OH, OW, W, 6, 6, deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
			else f88s1(stream, 3, ic_index, j_index, deltaY, OH, OW, W, 6, 6, deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
		}
		//======[FH = FW = 7]================================================================================================================================
		else if ((FH == 7) && (FW == 7)) {
			if (!(IH & 3) && !(IW & 3)) f88s1x4(stream, 3, ic_index, j_index, deltaY, OH, OW, W, 7, 7, deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
			else f88s1(stream, 3, ic_index, j_index, deltaY, OH, OW, W, 7, 7, deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
		}
		//======[Common]=====================================================================================================================================
		else if (!(N & 7 && !(GM & 63) && (j_index == 0))) {//OC % 8 == 0 && N % 8 == 0
			if (IS_POWER2(OC)) k88As1_oc2pow(stream, 3, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, LOG2(OC), ph, pw, GN, GM);
			else k88As1(stream, 3, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, OC, ph, pw, GN, GM);
		}
		else k88s1(stream, 3, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
		dconv3d_dX_ZeroPadding_s1_Branch(63, 63); return;
	}

	//==========================================[Main Part: [32-64, 32-64], GK % 4 == 0]======================================================
	if ((GN > 63) && (GM > 31)) {//[64, 32]
		if(!(OC & 7)) u84s1_pure(stream, 3, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
		else k84s1_pure(stream, 3, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
		dconv3d_dX_ZeroPadding_s1_Branch(63, 31); return;
	}
	if ((GN > 31) && (GM > 63)) {//[32, 64]
		if(!(OC & 7)) u48s1_pure(stream, 3, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
		else k48s1_pure(stream, 3, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
		dconv3d_dX_ZeroPadding_s1_Branch(31, 63); return;
	}

	if ((GN > 31) && (GM > 31)) {//[32, 32]
		if (!(OC & 7)) u44s1_pure(stream, 3, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
		else k44s1_pure(stream, 3, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
		dconv3d_dX_ZeroPadding_s1_Branch(31, 31); return;
	}

	if ((GN > 63) && (GM > 15)) {//[64, 16]
		k82s1_pure(stream, 3, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
		dconv3d_dX_ZeroPadding_s1_Branch(63, 15); return;
	}
	if ((GN > 15) && (GM > 63)) {//[16, 64]
		k28s1_pure(stream, 3, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
		dconv3d_dX_ZeroPadding_s1_Branch(15, 63); return;
	}

	if ((GN > 31) && (GM > 15)) {//[32, 16]
		k42s1_pure(stream, 3, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
		dconv3d_dX_ZeroPadding_s1_Branch(31, 15); return;
	}
	if ((GN > 15) && (GM > 31)) {//[16, 32]
		k24s1_pure(stream, 3, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
		dconv3d_dX_ZeroPadding_s1_Branch(15, 31); return;
	}

	//==========================================[Small]=======================================================================================
	if (GK > 7) {//GK >= 8
		if ((GN > 15) && (GM > 15)) {//[16, 16]
			k22s1(stream, 3, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
			dconv3d_dX_ZeroPadding_s1_Branch(15, 15); return;
		}
		if ((GN > 15) && (GM > 7)) {//[16, 8]
			k21s1(stream, 3, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
			dconv3d_dX_ZeroPadding_s1_Branch(15, 7); return;
		}
		if ((GN > 7) && (GM > 15)) {//[8, 16]
			k12s1(stream, 3, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
			dconv3d_dX_ZeroPadding_s1_Branch(7, 15); return;
		}
		if ((GN > 7) && (GM > 7)) {//[8, 8]
			k11s1(stream, 3, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
			dconv3d_dX_ZeroPadding_s1_Branch(7, 7); return;
		}
	}

	if ((GN > 7) && (GM > 7)) {//[8, 8]
		k22s1(stream, 2, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
		dconv3d_dX_ZeroPadding_s1_Branch(7, 7); return;
	}
	if (GN > 7) {//[8, 4]
		k21s1(stream, 2, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
		dconv3d_dX_ZeroPadding_s1_Branch(7, 3); return;
	}
	if (GM > 7) {//[4, 8]
		k12s1(stream, 2, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
		dconv3d_dX_ZeroPadding_s1_Branch(3, 7); return;
	}
	k11s1(stream, 2, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, ph, pw, GN, GM);//[4, 4]
}

#endif


//Dense Kernel: for sh = sw = 1, with winograd_F(2, 3)
#ifndef DECONV_3D_DELTAX_ZERO_PADDING_S1_TEXTURE
#define DECONV_3D_DELTAX_ZERO_PADDING_S1_TEXTURE

#ifndef DECONV_3D_DELTAX_ZERO_PADDING_S1_TEXTURE_MICRO
#define DECONV_3D_DELTAX_ZERO_PADDING_S1_TEXTURE_MICRO

#define __dconv3D_deltaX_ZeroPadding_s1_tex(streams, index, length, texDy, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, OC, ph, pw) \
	dconv3d_deltaX_ZeroPadding_s1_texture(streams,index,length, texDy,deltaY,OH,OW, W,FH,FW, deltaX,IH,IW, N,IC,OC, ph,pw,\
		GET_GN_ZeroPadding(IC), GET_GM_ZeroPadding(N,IH,IW), GET_GK_ZeroPadding(OC,FH,FW), 0, 0)

#define dconv3d_dX_ZeroPadding_s1_Branch_tex(SIZE_Y, SIZE_X) {\
	int GNr = GN & (SIZE_Y), GMr = GM & (SIZE_X);\
	if(GNr && GMr){\
		int next_ic_index = (GN - GNr) + ic_index;\
		int next_j_index = (GM - GMr) + j_index;\
		dconv3d_deltaX_ZeroPadding_s1_texture(streams,index,length, texDy,deltaY,OH,OW, W,FH,FW, deltaX,IH,IW, N,IC,OC, ph,pw,\
			GNr, GM, GK, next_ic_index, j_index);\
		dconv3d_deltaX_ZeroPadding_s1_texture(streams,index,length, texDy,deltaY,OH,OW, W,FH,FW, deltaX,IH,IW, N,IC,OC, ph,pw,\
			GN, GMr, GK, ic_index, next_j_index);\
		dconv3d_deltaX_ZeroPadding_s1_texture(streams,index,length, texDy,deltaY,OH,OW, W,FH,FW, deltaX,IH,IW, N,IC,OC, ph,pw,\
			GNr, GMr, GK, next_ic_index, next_j_index);}\
	else if(GNr){\
		int next_ic_index = (GN - GNr) + ic_index;\
		dconv3d_deltaX_ZeroPadding_s1_texture(streams,index,length, texDy,deltaY,OH,OW, W,FH,FW, deltaX,IH,IW, N,IC,OC, ph,pw,\
			GNr, GM, GK, next_ic_index, j_index);}\
	else if(GMr){\
		int next_j_index = (GM - GMr) + j_index;\
		dconv3d_deltaX_ZeroPadding_s1_texture(streams,index,length, texDy,deltaY,OH,OW, W,FH,FW, deltaX,IH,IW, N,IC,OC, ph,pw,\
			GN, GMr, GK, ic_index, next_j_index);}}

#endif

//(1) FH*FW >= 2
//(2) GN = IC;           GN >= 4, GN%4 == 0
//(3) GM = N  * IH * IW; GM >= 4, GM%4 == 0
//(4) GK = OC * FH * FW; GK >= 8, GK%4 == 0
//(5) sh = 1, sw = 1
void dconv3d_deltaX_ZeroPadding_s1_texture(jlong *streams, int &index, int length,
	cudaTextureObject_t texDy, const float* deltaY, int OH, int OW,
	const float*            W, int FH, int FW, 
	      float*       deltaX, int IH, int IW,
	int N, int IC, int OC,
	int ph, int pw,
	int GN, int GM, int GK,
	int ic_index, int j_index)
{
	next_cudaStream(stream, streams, index, length);

	if ((FH == 3) && (FW == 3) && (GN > 63) && (GM > 127)) {//Im2col-Winograd: [64, 128]
		bool flag = deconv3D_dX_winograd_f2x3_k48_tex<4>(stream, ic_index, j_index, texDy, OH, OW, W, deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
		if (flag) { dconv3d_dX_ZeroPadding_s1_Branch_tex(63, 127); return; }//[64, 128]
	}

	//==========================================[Main Part: [128, 128], GK % 8 == 0]=========================================================================
	if ((GN > 127) && (GM > 127) && !(GK & 7)) {
		if (deconv3D_dX_ZeroPadding_s1_uernel_ruse_8x8<4>(stream, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, ph, pw, GN, GM));//ruse uernels
		else if (deconv3D_dX_ZeroPadding_s1_uernel_8x8<4>(stream, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, OC, ph, pw, GN, GM));//common uernels
		else if (IS_POWER2(FH) && IS_POWER2(FW)) k88s1_W2pow(stream, 4, ic_index, j_index, deltaY, OH, OW, W, LOG2(FH), LOG2(FW), deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
		//======[FH = FW = 3]================================================================================================================================
		else if ((FH == 3) && (FW == 3)) {
			if (IS_POWER2(OC)) {
				if (!(IH & 3) && !(IW & 3)) k88s1W3x4_oc2pow_tex(stream, 4, ic_index, j_index, texDy, OH, OW, W, deltaX, IH, IW, IC, LOG2(OC), ph, pw, GN, GM);
				else if (!(GM & 127) && (j_index == 0) && !(N & 7) && !(OC & 7))  k88As1W3_oc2pow_tex(stream, 4, ic_index, j_index, texDy, OH, OW, W, deltaX, IH, IW, N, IC, LOG2(OC), ph, pw, GN, GM);
				else k88s1W3_oc2pow(stream, 4, ic_index, j_index, deltaY, OH, OW, W, deltaX, IH, IW, IC, LOG2(OC), ph, pw, GN, GM);
			}
			else if (!(GM & 127) && (j_index == 0) && !(N & 7) && !(OC & 7)) k88As1W3(stream, 4, ic_index, j_index, deltaY, OH, OW, W, deltaX, IH, IW, N, IC, OC, ph, pw, GN, GM);
			else if (!(IH & 3) && !(IW & 3)) f88s1x4_tex(stream, 4, ic_index, j_index, texDy, OH, OW, W, 3, 3, deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
			else f88s1_tex(stream, 4, ic_index, j_index, texDy, OH, OW, W, 3, 3, deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
		}
		//======[FH = FW = 5]================================================================================================================================
		else if ((FH == 5) && (FW == 5)) {
			if (IS_POWER2(OC)) {
				if (!(IH & 3) && !(IW & 3)) k88s1W5x4_oc2pow_tex(stream, 4, ic_index, j_index, texDy, OH, OW, W, deltaX, IH, IW, IC, LOG2(OC), ph, pw, GN, GM);
				else if (!(GM & 127) && (j_index == 0) && !(N & 7) && !(OC & 7)) k88As1W5_oc2pow_tex(stream, 4, ic_index, j_index, texDy, OH, OW, W, deltaX, IH, IW, N, IC, LOG2(OC), ph, pw, GN, GM);
				else k88s1W5_oc2pow(stream, 4, ic_index, j_index, deltaY, OH, OW, W, deltaX, IH, IW, IC, LOG2(OC), ph, pw, GN, GM);
			}
			else if (!(GM & 127) && (j_index == 0) && !(N & 7) && !(OC & 7)) k88As1W5(stream, 4, ic_index, j_index, deltaY, OH, OW, W, deltaX, IH, IW, N, IC, OC, ph, pw, GN, GM);
			else if (!(IH & 3) && !(IW & 3)) f88s1x4_tex(stream, 4, ic_index, j_index, texDy, OH, OW, W, 5, 5, deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
			else f88s1_tex(stream, 4, ic_index, j_index, texDy, OH, OW, W, 5, 5, deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
		}
		//======[FH = FW = 6]================================================================================================================================
		else if ((FH == 6) && (FW == 6)) {
			if (!(IH & 3) && !(IW & 3)) f88s1x4_tex(stream, 4, ic_index, j_index, texDy, OH, OW, W, 6, 6, deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
			else f88s1_tex(stream, 4, ic_index, j_index, texDy, OH, OW, W, 6, 6, deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
		}
		//======[FH = FW = 7]================================================================================================================================
		else if ((FH == 7) && (FW == 7)) {
			if (!(IH & 3) && !(IW & 3)) f88s1x4_tex(stream, 4, ic_index, j_index, texDy, OH, OW, W, 7, 7, deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
			else f88s1_tex(stream, 4, ic_index, j_index, texDy, OH, OW, W, 7, 7, deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
		}
		//======[Common]=====================================================================================================================================
		else if (!(N & 7) && !(OC & 7) && !(GM & 127) && (j_index == 0)) {//OC % 8 == 0 && N % 8 == 0
			if (IS_POWER2(OC)) k88As1_oc2pow_tex(stream, 4, ic_index, j_index, texDy, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, LOG2(OC), ph, pw, GN, GM);
			else k88As1_tex(stream, 4, ic_index, j_index, texDy, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, OC, ph, pw, GN, GM);
		}
		else k88s1(stream, 4, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
		dconv3d_dX_ZeroPadding_s1_Branch_tex(127, 127); return;
	}

	//==========================================[Main Part: [64, 64], GK % 4 == 0]===========================================================================
	if ((GN > 63) && (GM > 63)) {
		if (deconv3D_dX_ZeroPadding_s1_uernel_ruse_8x8<3>(stream, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, ph, pw, GN, GM));//ruse uernels
		else if (deconv3D_dX_ZeroPadding_s1_uernel_8x8<3>(stream, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, OC, ph, pw, GN, GM));//common uernels
		else if (IS_POWER2(FH) && IS_POWER2(FW)) k88s1_W2pow(stream, 3, ic_index, j_index, deltaY, OH, OW, W, LOG2(FH), LOG2(FW), deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
		//======[FH = FW = 3]================================================================================================================================
		else if ((FH == 3) && (FW == 3)) {
			if (IS_POWER2(OC)) {
				if (!(IH & 3) && !(IW & 3)) k88s1W3x4_oc2pow_tex(stream, 3, ic_index, j_index, texDy, OH, OW, W, deltaX, IH, IW, IC, LOG2(OC), ph, pw, GN, GM);
				else if (!(GM & 127) && (j_index == 0) && !(N & 7) && !(OC & 7))  k88As1W3_oc2pow_tex(stream, 3, ic_index, j_index, texDy, OH, OW, W, deltaX, IH, IW, N, IC, LOG2(OC), ph, pw, GN, GM);
				else k88As1W3_oc2pow_tex(stream, 3, ic_index, j_index, texDy, OH, OW, W, deltaX, IH, IW, N, IC, LOG2(OC), ph, pw, GN, GM);
			}
			else if (!(GM & 127) && (j_index == 0) && !(N & 7) && !(OC & 7)) k88As1W3(stream, 3, ic_index, j_index, deltaY, OH, OW, W, deltaX, IH, IW, N, IC, OC, ph, pw, GN, GM);
			else if (!(IH & 3) && !(IW & 3)) f88s1x4_tex(stream, 3, ic_index, j_index, texDy, OH, OW, W, 3, 3, deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
			else f88s1_tex(stream, 3, ic_index, j_index, texDy, OH, OW, W, 3, 3, deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
		}
		//======[FH = FW = 5]================================================================================================================================
		else if ((FH == 5) && (FW == 5)) {
			if (IS_POWER2(OC)) {
				if (!(IH & 3) && !(IW & 3)) k88s1W5x4_oc2pow_tex(stream, 3, ic_index, j_index, texDy, OH, OW, W, deltaX, IH, IW, IC, LOG2(OC), ph, pw, GN, GM);
				else if (!(GM & 127) && (j_index == 0) && !(N & 7) && !(OC & 7)) k88As1W5_oc2pow_tex(stream, 3, ic_index, j_index, texDy, OH, OW, W, deltaX, IH, IW, N, IC, LOG2(OC), ph, pw, GN, GM);
				else k88As1W5_oc2pow_tex(stream, 3, ic_index, j_index, texDy, OH, OW, W, deltaX, IH, IW, N, IC, LOG2(OC), ph, pw, GN, GM);
			}
			else if (!(GM & 127) && (j_index == 0) && !(N & 7) && !(OC & 7)) k88As1W5(stream, 3, ic_index, j_index, deltaY, OH, OW, W, deltaX, IH, IW, N, IC, OC, ph, pw, GN, GM);
			else if (!(IH & 3) && !(IW & 3)) f88s1x4_tex(stream, 3, ic_index, j_index, texDy, OH, OW, W, 5, 5, deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
			else f88s1_tex(stream, 3, ic_index, j_index, texDy, OH, OW, W, 5, 5, deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
		}
		//======[FH = FW = 6]================================================================================================================================
		else if ((FH == 6) && (FW == 6)) {
			if (!(IH & 3) && !(IW & 3)) f88s1x4_tex(stream, 3, ic_index, j_index, texDy, OH, OW, W, 6, 6, deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
			else f88s1_tex(stream, 3, ic_index, j_index, texDy, OH, OW, W, 6, 6, deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
		}
		//======[FH = FW = 7]================================================================================================================================
		else if ((FH == 7) && (FW == 7)) {
			if (!(IH & 3) && !(IW & 3)) f88s1x4_tex(stream, 3, ic_index, j_index, texDy, OH, OW, W, 7, 7, deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
			else f88s1_tex(stream, 3, ic_index, j_index, texDy, OH, OW, W, 7, 7, deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
		}
		//======[Common]=====================================================================================================================================
		else if (!(N & 7) && !(GM & 63) && (j_index == 0)) {//OC % 8 == 0 && N % 8 == 0
			if (IS_POWER2(OC)) k88As1_oc2pow_tex(stream, 3, ic_index, j_index, texDy, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, LOG2(OC), ph, pw, GN, GM);
			else k88As1_tex(stream, 3, ic_index, j_index, texDy, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, OC, ph, pw, GN, GM);
		}
		else k88s1_tex(stream, 3, ic_index, j_index, texDy, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
		dconv3d_dX_ZeroPadding_s1_Branch_tex(63, 63); return;
	}

	//==========================================[Main Part: [32-64, 32-64], GK % 4 == 0]=====================================================================
	if ((GN > 63) && (GM > 31)) {//[64, 32]
		if(!(OC & 7)) u84s1_pure(stream, 3, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
		else k84s1_pure(stream, 3, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
		dconv3d_dX_ZeroPadding_s1_Branch(63, 31); return;
	}

	if ((GN > 31) && (GM > 63)) {//[32, 64]
		if ((FH == 3) && (FW == 3) && deconv3D_dX_winograd_f2x3_k48_tex<3>(stream, ic_index, j_index, texDy, OH, OW, W, deltaX, IH, IW, IC, OC, ph, pw, GN, GM));
		else if (!(OC & 7)) u48s1_pure(stream, 3, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
		else k48s1_pure_tex(stream, 3, ic_index, j_index, texDy, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
		dconv3d_dX_ZeroPadding_s1_Branch_tex(31, 63); return;
	}

	if ((GN > 31) && (GM > 31)) {//[32, 32], GK % 4 == 0
		if (!(OC & 7)) u44s1_pure(stream, 3, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
		else k44s1_pure_tex(stream, 3, ic_index, j_index, texDy, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
		dconv3d_dX_ZeroPadding_s1_Branch(31, 31); return;
	}

	if ((GN > 63) && (GM > 15)) {//[64, 16]
		k82s1_pure(stream, 3, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
		dconv3d_dX_ZeroPadding_s1_Branch(63, 15); return;
	}
	if ((GN > 15) && (GM > 63)) {//[16, 64]
		k28s1_pure(stream, 3, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
		dconv3d_dX_ZeroPadding_s1_Branch(15, 63); return;
	}

	if ((GN > 31) && (GM > 15)) {//[32, 16], GK % 4 == 0
		k42s1_pure(stream, 3, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
		dconv3d_dX_ZeroPadding_s1_Branch(31, 15); return;
	}

	if ((GN > 15) && (GM > 31)) {//[16, 32], GK % 4 == 0
		k24s1_pure(stream, 3, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
		dconv3d_dX_ZeroPadding_s1_Branch(15, 31); return;
	}

	//==========================================[Small]======================================================================================================
	if (GK > 7) {//GK >= 8
		if ((GN > 15) && (GM > 15)) {//[16, 16]
			k22s1(stream, 3, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
			dconv3d_dX_ZeroPadding_s1_Branch(15, 15); return;
		}
		if ((GN > 15) && (GM > 7)) {//[16, 8]
			k21s1(stream, 3, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
			dconv3d_dX_ZeroPadding_s1_Branch(15, 7); return;
		}
		if ((GN > 7) && (GM > 15)) {//[8, 16]
			k12s1(stream, 3, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
			dconv3d_dX_ZeroPadding_s1_Branch(7, 15); return;
		}
		if ((GN > 7) && (GM > 7)) {//[8, 8]
			k11s1(stream, 3, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
			dconv3d_dX_ZeroPadding_s1_Branch(7, 7); return;
		}
	}

	if ((GN > 7) && (GM > 7)) {//[8, 8]
		k22s1(stream, 2, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
		dconv3d_dX_ZeroPadding_s1_Branch(7, 7); return;
	}
	if (GN > 7) {//[8, 4]
		k21s1(stream, 2, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
		dconv3d_dX_ZeroPadding_s1_Branch(7, 3); return;
	}
	if (GM > 7) {//[4, 8]
		k12s1(stream, 2, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
		dconv3d_dX_ZeroPadding_s1_Branch(3, 7); return;
	}
	k11s1(stream, 2, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
}

#endif


//Dense Kernel: for sh = sw = 1
#ifndef DECONV_3D_DELTAX_ZERO_PADDING_V2_S1
#define DECONV_3D_DELTAX_ZERO_PADDING_V2_S1

#ifndef DECONV_3D_DELTAX_ZERO_PADDING_V2_S1_MICRO
#define DECONV_3D_DELTAX_ZERO_PADDING_V2_S1_MICRO

#define __dconv3D_deltaX_ZeroPaddingV2_s1(env,streams, index, length, deltaY, OH, OW, sizeY, W, FH, FW, deltaX, IH, IW, N, IC, OC, ph, pw) \
	dconv3d_deltaX_ZeroPaddingV2_s1(env,streams,index,length,\
		deltaY,OH,OW,sizeY, W,FH,FW, deltaX,IH,IW, IC,OC, ph,pw,\
		IC, N, 0, 0)

#define dconv3d_dX_ZeroPaddingV2_s1_Branch(SIZE_Y, SIZE_X) {\
	int GNr = GN & (SIZE_Y), Nr = N & (SIZE_X);\
	if(GNr && Nr){\
		int next_ic_index = (GN - GNr) + ic_index;\
		int next_n_index = (N - Nr) + n_index;\
		dconv3d_deltaX_ZeroPaddingV2_s1(env,streams,index,length, deltaY,OH,OW,sizeY, W,FH,FW, deltaX,IH,IW, IC,OC, ph,pw,\
			GNr, N, next_ic_index, n_index);\
		dconv3d_deltaX_ZeroPaddingV2_s1(env,streams,index,length, deltaY,OH,OW,sizeY, W,FH,FW, deltaX,IH,IW, IC,OC, ph,pw,\
			GN, Nr, ic_index, next_n_index);\
		dconv3d_deltaX_ZeroPaddingV2_s1(env,streams,index,length, deltaY,OH,OW,sizeY, W,FH,FW, deltaX,IH,IW, IC,OC, ph,pw,\
			GNr, Nr, next_ic_index, next_n_index);}\
	else if(GNr){\
		int next_ic_index = (GN - GNr) + ic_index;\
		dconv3d_deltaX_ZeroPaddingV2_s1(env,streams,index,length, deltaY,OH,OW,sizeY, W,FH,FW, deltaX,IH,IW, IC,OC, ph,pw,\
			GNr, N, next_ic_index, n_index);}\
	else if(Nr){\
		int next_n_index = (N - Nr) + n_index;\
		dconv3d_deltaX_ZeroPaddingV2_s1(env,streams,index,length, deltaY,OH,OW,sizeY, W,FH,FW, deltaX,IH,IW, IC,OC, ph,pw,\
			GN, Nr, ic_index, next_n_index);}}

#endif

//(1) FH*FW >= 2
//(2) GN = IC;           GN >= 4, GN%4 == 0
//(3) GM = N;            GM >= 4, GM%4 == 0
//(4) GK = OC * FH * FW; GK >= 8, GK%4 == 0
//(5) sh = 1, sw = 1
void dconv3d_deltaX_ZeroPaddingV2_s1(JNIEnv *env, jlong *streams, int &index, int length,
	const float* deltaY, int OH, int OW, int sizeY,
	const float*      W, int FH, int FW,
	      float* deltaX, int IH, int IW,
	int IC, int OC,
	int ph, int pw,
	int GN, int N,
	int ic_index, int n_index)
{
	next_cudaStream(stream, streams, index, length);

	//======[Stage1]===========================================================================================================================
	if ((GN > 127) && (N > 127) && !(OC & 7)) {//[128, 128], OC % 8 == 0
		//======[FH = FW = 3, ph = pw = 1, (OH, OW) >= 2]======================================================================================
		if (CAN_s1_V2_W3P1) {
			if (IS_POWER2(OC)) {
				if (!(OC & 15)) uV2_88s1W3P1_oc2pow(stream, 4, ic_index, n_index, deltaY, OH, OW, W, deltaX, IH, IW, IC, LOG2(OC), GN, N);
				else kV2_88s1W3P1_oc2pow(stream, 4, ic_index, n_index, deltaY, OH, OW, W, deltaX, IH, IW, IC, LOG2(OC), GN, N);
			}
			else if (!(OC & 15)) uV2_88s1W3P1(stream, 4, ic_index, n_index, deltaY, OH, OW, W, deltaX, IH, IW, IC, OC, GN, N);
			else kV2_88s1W3P1(stream, 4, ic_index, n_index, deltaY, OH, OW, W, deltaX, IH, IW, IC, OC, GN, N);
		}
		//======[FH = FW = 5, ph = pw = 2, (OH, OW) >= 3]======================================================================================
		else if (CAN_s1_V2_W5P2) {
			if (IS_POWER2(OC)) {
				if (!(OC & 15)) uV2_88s1W5P2_oc2pow(stream, 4, ic_index, n_index, deltaY, OH, OW, W, deltaX, IH, IW, IC, LOG2(OC), GN, N);
				else kV2_88s1W5P2_oc2pow(stream, 4, ic_index, n_index, deltaY, OH, OW, W, deltaX, IH, IW, IC, LOG2(OC), GN, N);
			}
			else if (!(OC & 15)) uV2_88s1W5P2(stream, 4, ic_index, n_index, deltaY, OH, OW, W, deltaX, IH, IW, IC, OC, GN, N);
			else kV2_88s1W5P2(stream, 4, ic_index, n_index, deltaY, OH, OW, W, deltaX, IH, IW, IC, OC, GN, N);
		}
		//======[Common]=======================================================================================================================
		else if (IS_POWER2(OC)) {
			if (!(OC & 15)) uV2_88s1_oc2pow(stream, 4, ic_index, n_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, LOG2(OC), ph, pw, GN, N);
			else kV2_88s1_oc2pow(stream, 4, ic_index, n_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, LOG2(OC), ph, pw, GN, N);
		}
		else if (!(OC & 15)) uV2_88s1(stream, 4, ic_index, n_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, ph, pw, GN, N);
		else kV2_88s1(stream, 4, ic_index, n_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, ph, pw, GN, N);
		dconv3d_dX_ZeroPaddingV2_s1_Branch(127, 127); return;
	}

	float scale_up = s1_PADDING_SCALE_UP(IH, IW, OH, OW, FH, FW);
	if (scale_up < 1.3f) {//gate1
		s1_V2_TO_V1(FH, FW, IH, IW, N, OC, n_index);
		if (index > 0) index = index - 1;

		if (env) {//env != null, use texture
			cudaTextureObject_t texDy = floatTexture((float*)deltaY, sizeY, env);
			dconv3d_deltaX_ZeroPadding_s1_texture(streams, index, length,
				texDy, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, OC, ph, pw,
				GN, GM, GK, ic_index, j_index);
			cudaError_t error = cudaDestroyTextureObject(texDy); handleError(error);
			return;
		}

		dconv3d_deltaX_ZeroPadding_s1(streams, index, length,
			deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, OC, ph, pw,
			GN, GM, GK, ic_index, j_index);
		return;
	}

	//======[Stage2]===========================================================================================================================
	if ((GN > 63) && (N > 63)) {//[64, 64], OC % 4 == 0
		//======[FH = FW = 3, ph = pw = 1, (OH, OW) >= 2]======================================================================================
		if (CAN_s1_V2_W3P1) {
			if (IS_POWER2(OC)) {
				if (!(OC & 7)) uV2_88s1W3P1_oc2pow(stream, 3, ic_index, n_index, deltaY, OH, OW, W, deltaX, IH, IW, IC, LOG2(OC), GN, N);
				else kV2_88s1W3P1_oc2pow(stream, 3, ic_index, n_index, deltaY, OH, OW, W, deltaX, IH, IW, IC, LOG2(OC), GN, N);
			}
			else if (!(OC & 7)) uV2_88s1W3P1(stream, 3, ic_index, n_index, deltaY, OH, OW, W, deltaX, IH, IW, IC, OC, GN, N);
			else kV2_88s1W3P1(stream, 3, ic_index, n_index, deltaY, OH, OW, W, deltaX, IH, IW, IC, OC, GN, N);
		}
		//======[FH = FW = 5, ph = pw = 2, (OH, OW) >= 3]======================================================================================
		else if (CAN_s1_V2_W5P2) {
			if (IS_POWER2(OC)) {
				if (!(OC & 7)) uV2_88s1W5P2_oc2pow(stream, 3, ic_index, n_index, deltaY, OH, OW, W, deltaX, IH, IW, IC, LOG2(OC), GN, N);
				else kV2_88s1W5P2_oc2pow(stream, 3, ic_index, n_index, deltaY, OH, OW, W, deltaX, IH, IW, IC, LOG2(OC), GN, N);
			}
			else if (!(OC & 7)) uV2_88s1W5P2(stream, 3, ic_index, n_index, deltaY, OH, OW, W, deltaX, IH, IW, IC, OC, GN, N);
			else kV2_88s1W5P2(stream, 3, ic_index, n_index, deltaY, OH, OW, W, deltaX, IH, IW, IC, OC, GN, N);
		}
		//======[Common]=======================================================================================================================
		else if (IS_POWER2(OC)) {
			if (!(OC & 7)) uV2_88s1_oc2pow(stream, 3, ic_index, n_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, LOG2(OC), ph, pw, GN, N);
			else kV2_88s1_oc2pow(stream, 3, ic_index, n_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, LOG2(OC), ph, pw, GN, N);
		}
		else if (!(OC & 7)) uV2_88s1(stream, 3, ic_index, n_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, ph, pw, GN, N);
		else kV2_88s1(stream, 3, ic_index, n_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, ph, pw, GN, N);
		dconv3d_dX_ZeroPaddingV2_s1_Branch(63, 63); return;
	}

	if (scale_up < 1.4f) {//gate2
		s1_V2_TO_V1(FH, FW, IH, IW, N, OC, n_index);
		if (index > 0) index = index - 1;

		if (env) {//env != null, use texture
			cudaTextureObject_t texDy = floatTexture((float*)deltaY, sizeY, env);
			dconv3d_deltaX_ZeroPadding_s1_texture(streams, index, length,
				texDy, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, OC, ph, pw, 
				GN, GM, GK, ic_index, j_index);
			cudaError_t error = cudaDestroyTextureObject(texDy); handleError(error);
			return;
		}

		dconv3d_deltaX_ZeroPadding_s1(streams, index, length,
			deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, OC, ph, pw,
			GN, GM, GK, ic_index, j_index);
		return;
	}

	//======[Stage3]===========================================================================================================================
	if ((GN > 63) && (N > 31)) {
		if (IS_POWER2(OC)) kV2_84s1_oc2pow(stream, 3, ic_index, n_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, LOG2(OC), ph, pw, GN, N);
		else kV2_84s1(stream, 3, ic_index, n_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, ph, pw, GN, N);
		dconv3d_dX_ZeroPaddingV2_s1_Branch(63, 31); return;
	}
	if ((GN > 31) && (N > 63)) {
		if (IS_POWER2(OC)) kV2_48s1_oc2pow(stream, 3, ic_index, n_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, LOG2(OC), ph, pw, GN, N);
		else kV2_48s1(stream, 3, ic_index, n_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, ph, pw, GN, N);
		dconv3d_dX_ZeroPaddingV2_s1_Branch(31, 63); return;
	}

	if ((GN > 31) && (N > 31)) {
		if (IS_POWER2(OC)) kV2_44s1_oc2pow(stream, 3, ic_index, n_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, LOG2(OC), ph, pw, GN, N);
		else kV2_44s1(stream, 3, ic_index, n_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, ph, pw, GN, N);
		dconv3d_dX_ZeroPaddingV2_s1_Branch(31, 31); return;
	}

	s1_V2_TO_V1(FH, FW, IH, IW, N, OC, n_index);//gate3
	if (index > 0) index = index - 1;

	if (env) {//env != null, use texture
		cudaTextureObject_t texDy = floatTexture((float*)deltaY, sizeY, env);
		dconv3d_deltaX_ZeroPadding_s1_texture(streams, index, length,
			texDy, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, OC, ph, pw,
			GN, GM, GK, ic_index, j_index);
		cudaError_t error = cudaDestroyTextureObject(texDy); handleError(error);
		return;
	}

	dconv3d_deltaX_ZeroPadding_s1(streams, index, length,
		deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, OC, ph, pw,
		GN, GM, GK, ic_index, j_index);
}

#endif


//Dense Kernel : sh = sw = 1, FH = FW = 1, ph = pw = 1
#ifndef DECONV_3D_DELTAX_W1
#define DECONV_3D_DELTAX_W1

#ifndef DECONV_3D_DELTAX_W1_MICRO
#define DECONV_3D_DELTAX_W1_MICRO

#define __dconv3D_deltaX_W1(streams, index, length, deltaY, W, deltaX,IH,IW, N,IC,OC) \
	dconv3d_deltaX_W1(streams, index, length, deltaY, W, deltaX, IC,OC,\
		GET_GN_ZeroPadding(IC), GET_GM_ZeroPadding(N,IH,IW), 0, 0)\

#define dconv3d_dX_W1_Branch(SIZE_Y, SIZE_X) {\
	int GNr = GN & (SIZE_Y), GMr = GM & (SIZE_X);\
	if(GNr && GMr){\
		int next_ic_index = (GN - GNr) + ic_index;\
		int next_j_index = (GM - GMr) + j_index;\
		dconv3d_deltaX_W1(streams, index, length, deltaY, W, deltaX, IC,OC,\
			GNr, GM, next_ic_index, j_index);\
		dconv3d_deltaX_W1(streams, index, length, deltaY, W, deltaX, IC,OC,\
			GN, GMr, ic_index, next_j_index);\
		dconv3d_deltaX_W1(streams, index, length, deltaY, W, deltaX, IC,OC,\
			GNr, GMr, next_ic_index, next_j_index);}\
	else if(GNr){\
		int next_ic_index = (GN - GNr) + ic_index;\
		dconv3d_deltaX_W1(streams, index, length, deltaY, W, deltaX, IC,OC,\
			GNr, GM, next_ic_index, j_index);}\
	else if(GMr){\
		int next_j_index = (GM - GMr) + j_index;\
		dconv3d_deltaX_W1(streams, index, length, deltaY, W, deltaX, IC,OC,\
			GN, GMr, ic_index, next_j_index);}}

#endif

//(1) FH ==1, FW == 1
//(2) GN = IC;           GN >= 4, GN%4 == 0
//(3) GM = N  * IH * IW; GM >= 4, GM%4 == 0
//(4) GK = OC;           GK >= 4, GK%4 == 0
//(5) sh = 1, sw = 1
//(6) ph = 0, pw = 0
void dconv3d_deltaX_W1(jlong *streams, int &index, int length,
	const float* deltaY,
	const float*      W,//FH = FW = 1
		  float* deltaX, 
	int IC, int OC,
	int GN, int GM, 
	int ic_index, int j_index)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)streams[index];
	index = (index + 1) % length;

	if ((GN > 127) && (GM > 127) && !(OC & 7)) {//[128, 128], GK = OC % 8 == 0
		k88W1_LB4(stream, ic_index, j_index, deltaY, W, deltaX, IC, OC, GN, GM);
		dconv3d_dX_W1_Branch(127, 127); return;
	}
	if ((GN > 63) && (GM > 127) && !(OC & 7)) {//[64, 128], GK = OC % 8 == 0
		k48W1(stream, 4, ic_index, j_index, deltaY, W, deltaX, IC, OC, GN, GM);
		dconv3d_dX_W1_Branch(63, 127); return;
	}
	if ((GN > 63) && (GM > 63)) {//[64, 64]
		k88W1_LB3(stream, ic_index, j_index, deltaY, W, deltaX, IC, OC, GN, GM);
		dconv3d_dX_W1_Branch(63, 63); return;
	}

	if ((GN > 31) && (GM > 63)) {//[32, 64]
		k48W1(stream, 3, ic_index, j_index, deltaY, W, deltaX, IC, OC, GN, GM);
		dconv3d_dX_W1_Branch(31, 63); return;
	}
	if ((GN > 63) && (GM > 31)) {//[64, 32]
		k84W1(stream, 3, ic_index, j_index, deltaY, W, deltaX, IC, OC, GN, GM);
		dconv3d_dX_W1_Branch(63, 31); return;
	}
 
	if ((GN > 31) && (GM > 31)) {//[32, 32]
		k44W1(stream, 3, ic_index, j_index, deltaY, W, deltaX, IC, OC, GN, GM);
		dconv3d_dX_W1_Branch(31, 31); return;
	}

	if ((GN > 63) && (GM > 15)) {//[64,1 6]
		k82W1(stream, 3, ic_index, j_index, deltaY, W, deltaX, IC, OC, GN, GM);
		dconv3d_dX_W1_Branch(63, 15); return;
	}
	if ((GN > 15) && (GM > 63)) {//[16, 64]
		k28W1(stream, 3, ic_index, j_index, deltaY, W, deltaX, IC, OC, GN, GM);
		dconv3d_dX_W1_Branch(15, 63); return;
	}

	if ((GN > 31) && (GM > 15)) {//[32, 16], OC % 4 == 0
		k42W1(stream, 3, ic_index, j_index, deltaY, W, deltaX, IC, OC, GN, GM);
		dconv3d_dX_W1_Branch(31, 15); return;
	}
	if ((GN > 15) && (GM > 31)) {//[16, 32], OC % 4 == 0
		k24W1(stream, 3, ic_index, j_index, deltaY, W, deltaX, IC, OC, GN, GM);
		dconv3d_dX_W1_Branch(15, 31); return;
	}

	if ((GN > 15) && (GM > 15)) {//[16, 16]
		k22W1(stream, 3, ic_index, j_index, deltaY, W, deltaX, IC, OC, GN, GM);
		dconv3d_dX_W1_Branch(15, 15); return;
	}
	if ((GN > 15) && (GM > 7)) {//[16, 8]
		k21W1(stream, 3, ic_index, j_index, deltaY, W, deltaX, IC, OC, GN, GM);
		dconv3d_dX_W1_Branch(15, 7); return;
	}
	if ((GN > 7) && (GM > 15)) {//[8, 16]
		k12W1(stream, 3, ic_index, j_index, deltaY, W, deltaX, IC, OC, GN, GM);
		dconv3d_dX_W1_Branch(7, 15); return;
	}
	if ((GN > 7) && (GM > 7)) {//[8, 8]
		k11W1(stream, 3, ic_index, j_index, deltaY, W, deltaX, IC, OC, GN, GM);
		dconv3d_dX_W1_Branch(7, 7); return;
	}

	if (GN > 7) {//[8, 4]
		k21W1(stream, 2, ic_index, j_index, deltaY, W, deltaX, IC, OC, GN, GM);
		dconv3d_dX_W1_Branch(7, 3); return;
	}
	if (GM > 7) {//[4, 8]
		k12W1(stream, 2, ic_index, j_index, deltaY, W, deltaX, IC, OC, GN, GM);
		dconv3d_dX_W1_Branch(3, 7); return;
	}
	k11W1(stream, 2, ic_index, j_index, deltaY, W, deltaX, IC, OC, GN, GM);
}

#endif

#endif



//kernel split: W[OC, FH, FW, IC] -> CW[sh, sw, OC, CFH, CFW, IC]
#ifndef KERNEL_SPLIT_AREA
#define KERNEL_SPLIT_AREA

//Sparse Kernel: for (IH, IW) % (sh, sw) != 0
#ifndef DCONV_3D_DELTAX_KERNEL_SPLIT_R
#define DCONV_3D_DELTAX_KERNEL_SPLIT_R

#ifndef DCONV_3D_DELTAX_KERNEL_SPLIT_R_MICRO
#define DCONV_3D_DELTAX_KERNEL_SPLIT_R_MICRO

#define __dconv3D_deltaX_ksR(streams,index,length, deltaY,OH,OW, CW,FH,FW,CWstride, deltaX,IH,IW, N,IC,OC, sh, sw, ph, pw) \
	dconv3D_deltaX_ksR(streams,index,length,\
		deltaY,OH,OW, CW,FH,FW,CWstride, deltaX,IH,IW, N,IC,OC, sh,sw,ph,pw,\
		GN, GM, 0,0)\

#define dconv3d_dX_ksR_Branch(SIZE_Y, SIZE_X) {\
	int GNr = GN & (SIZE_Y), GMr = GM & (SIZE_X);\
	if(GNr && GMr){\
		int next_ic_index = (GN - GNr) + ic_index;\
		int next_j_index = (GM - GMr) + j_index;\
		dconv3D_deltaX_ksR(streams,index,length, deltaY,OH,OW, CW,FH,FW,CWstride, deltaX,IH,IW, N,IC,OC, sh,sw,ph,pw,\
			GNr, GM,  next_ic_index, j_index);\
		dconv3D_deltaX_ksR(streams,index,length, deltaY,OH,OW, CW,FH,FW,CWstride, deltaX,IH,IW, N,IC,OC, sh,sw,ph,pw,\
			GN,  GMr, ic_index, next_j_index);\
		dconv3D_deltaX_ksR(streams,index,length, deltaY,OH,OW, CW,FH,FW,CWstride, deltaX,IH,IW, N,IC,OC, sh,sw,ph,pw,\
			GNr, GMr, next_ic_index, next_j_index);}\
	else if(GNr){\
		int next_ic_index = (GN - GNr) + ic_index;\
		dconv3D_deltaX_ksR(streams,index,length, deltaY,OH,OW, CW,FH,FW,CWstride, deltaX,IH,IW, N,IC,OC, sh,sw,ph,pw,\
			GNr, GM, next_ic_index, j_index);}\
	else if(GMr){\
		int next_j_index = (GM - GMr) + j_index;\
		dconv3D_deltaX_ksR(streams,index,length, deltaY,OH,OW, CW,FH,FW,CWstride, deltaX,IH,IW, N,IC,OC, sh,sw,ph,pw,\
			GN, GMr, ic_index, next_j_index);}}

#endif

//<1> (IH, IW) % (sh, sw) != 0
//<2> remode: W[OC, FH, FW, IC] -> CW[sh, sw, OC, CFH, CFW, IC]
//<3> OC % 4 == 0, OC >= 4
//<4> GM = N * IH_slice * IW_slice, GM % 4 == 0, GM >= 4
//<5> GN = IC, GN % 4 == 0, GN >= 4
void dconv3D_deltaX_ksR(jlong *streams, int &index, int length,
	const float* deltaY, int OH, int OW,
	const float*     CW, int FH, int FW, int CWstride,
	      float* deltaX, int IH, int IW,
	int N, int IC, int OC,
	int sh, int sw, int ph, int pw,
	int GN, int GM,
	int ic_index, int j_index)
{
	next_cudaStream(stream, streams, index, length);

	if ((GN > 127) && (GM > 127) && !(OC & 7)) {//OC % 8 == 0
		if (IS_POWER2(OC)) ks88R_oc2pow(stream, 4, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH, IW, N, IC, LOG2(OC), sh, sw, ph, pw, GN, GM);
		else ks88R(stream, 4, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, GN, GM);
		dconv3d_dX_ksR_Branch(127, 127); return;
	}
	if ((GN > 63) && (GM > 63)) {
		if (IS_POWER2(OC)) ks88R_oc2pow(stream, 3, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH, IW, N, IC, LOG2(OC), sh, sw, ph, pw, GN, GM);
		else ks88R(stream, 3, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, GN, GM);
		dconv3d_dX_ksR_Branch(63, 63); return;
	}

	if ((GN > 63) && (GM > 31)) {
		if (IS_POWER2(OC)) ks84R_oc2pow(stream, 3, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH, IW, N, IC, LOG2(OC), sh, sw, ph, pw, GN, GM);
		else ks84R(stream, 3, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, GN, GM);
		dconv3d_dX_ksR_Branch(31, 31); return;
	}

	if ((GN > 31) && (GM > 63)) {
		if (IS_POWER2(OC)) ks48R_oc2pow(stream, 3, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH, IW, N, IC, LOG2(OC), sh, sw, ph, pw, GN, GM);
		else ks48R(stream, 3, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, GN, GM);
		dconv3d_dX_ksR_Branch(31, 63); return;
	}

	if ((GN > 31) && (GM > 31)) {
		if (IS_POWER2(OC)) ks44R_oc2pow(stream, 3, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH, IW, N, IC, LOG2(OC), sh, sw, ph, pw, GN, GM);
		else ks44R(stream, 3, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, GN, GM);
		dconv3d_dX_ksR_Branch(31, 31); return;
	}

	if ((GN > 31) && (GM > 15)) {
		ks42R(stream, 3, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, GN, GM);
		dconv3d_dX_ksR_Branch(31, 15); return;
	}
	if ((GN > 15) && (GM > 31)) {
		ks24R(stream, 3, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, GN, GM);
		dconv3d_dX_ksR_Branch(15, 31); return;
	}

	if (OC > 7) {//BLOCK_SIZE = 8, OC >= 8
		if ((GN > 15) && (GM > 15)) {
			ks22R(stream, 3, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, GN, GM);
			dconv3d_dX_ksR_Branch(15, 15); return;
		}
		if ((GN > 15) && (GM > 7)) {
			ks21R(stream, 3, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, GN, GM);
			dconv3d_dX_ksR_Branch(15, 7); return;
		}
		if ((GN > 7) && (GM > 15)) {
			ks12R(stream, 3, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, GN, GM);
			dconv3d_dX_ksR_Branch(7, 15); return;
		}
		if ((GN > 7) && (GM > 7)) {
			ks11R(stream, 3, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, GN, GM);
			dconv3d_dX_ksR_Branch(7, 7); return;
		}
	}

	if ((GN > 7) && (GM > 7)) {
		ks22R(stream, 2, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, GN, GM);
		dconv3d_dX_ksR_Branch(7, 7); return;
	}
	if (GN > 7) {
		ks21R(stream, 2, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, GN, GM);
		dconv3d_dX_ksR_Branch(7, 3); return;
	}
	if (GM > 7) {
		ks12R(stream, 2, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, GN, GM);
		dconv3d_dX_ksR_Branch(3, 7); return;
	}
	ks11R(stream, 2, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, GN, GM);
}

#endif


//Sparse Kernel: for (IH, IW) % (sh, sw) = 0
#ifndef DCONV_3D_DELTAX_KERNEL_SPLIT_IMSR
#define DCONV_3D_DELTAX_KERNEL_SPLIT_IMSR

#ifndef DCONV_3D_DELTAX_KERNEL_SPLIT_IMSR_MICRO
#define DCONV_3D_DELTAX_KERNEL_SPLIT_IMSR_MICRO

#define __dconv3D_deltaX_ksImsR(streams,index,length, deltaY,OH,OW, CW,FH,FW,CWstride, deltaX,IH_slice,IW_slice, IC,OC, sh, sw, ph, pw) \
	dconv3D_deltaX_ksImsR(streams,index,length,\
		deltaY,OH,OW, CW,FH,FW,CWstride, deltaX,IH_slice,IW_slice, IC,OC, sh,sw,ph,pw,\
		GN, GM, 0,0)\

#define dconv3d_dX_ksImsR_Branch(SIZE_Y, SIZE_X) {\
	int GNr = GN & (SIZE_Y), GMr = GM & (SIZE_X);\
	if(GNr && GMr){\
		int next_ic_index = (GN - GNr) + ic_index;\
		int next_j_index = (GM - GMr) + j_index;\
		dconv3D_deltaX_ksImsR(streams,index,length, deltaY,OH,OW, CW,FH,FW,CWstride, deltaX,IH_slice,IW_slice, IC,OC, sh,sw,ph,pw,\
			GNr, GM,  next_ic_index, j_index);\
		dconv3D_deltaX_ksImsR(streams,index,length, deltaY,OH,OW, CW,FH,FW,CWstride, deltaX,IH_slice,IW_slice, IC,OC, sh,sw,ph,pw,\
			GN,  GMr, ic_index, next_j_index);\
		dconv3D_deltaX_ksImsR(streams,index,length, deltaY,OH,OW, CW,FH,FW,CWstride, deltaX,IH_slice,IW_slice, IC,OC, sh,sw,ph,pw,\
			GNr, GMr, next_ic_index, next_j_index);}\
	else if(GNr){\
		int next_ic_index = (GN - GNr) + ic_index;\
		dconv3D_deltaX_ksImsR(streams,index,length, deltaY,OH,OW, CW,FH,FW,CWstride, deltaX,IH_slice,IW_slice, IC,OC, sh,sw,ph,pw,\
			GNr, GM, next_ic_index, j_index);}\
	else if(GMr){\
		int next_j_index = (GM - GMr) + j_index;\
		dconv3D_deltaX_ksImsR(streams,index,length, deltaY,OH,OW, CW,FH,FW,CWstride, deltaX,IH_slice,IW_slice, IC,OC, sh,sw,ph,pw,\
			GN, GMr, ic_index, next_j_index);}}

#endif

//<1> IMS: INPUT_MOD_STEP:(IH, IW) % (sh, sw) == 0
//<2> remode: W[OC, FH, FW, IC] -> CW[sh, sw, OC, CFH, CFW, IC]
//<3> OC % 4 == 0, OC >= 4
//<4> GM = N * IH_slice * IW_slice, GM % 4 == 0, GM >= 4
//<5> GN = IC, GN % 4 == 0, GN >= 4
void dconv3D_deltaX_ksImsR(jlong *streams, int &index, int length,
	const float* deltaY, int OH, int OW,
	const float*     CW, int FH, int FW, int CWstride,
	      float* deltaX, int IH_slice, int IW_slice,
	int IC, int OC,
	int sh, int sw, int ph, int pw,
	int GN, int GM,
	int ic_index, int j_index)
{
	next_cudaStream(stream, streams, index, length);

	if (!(IH_slice & 7) && !(IW_slice & 7)) {//(IH_slice, IW_slice) % 8 == 0
		if ((GN > 127) && (GM > 127) && !(OC & 7)) {//OC % 8 == 0
			if (IS_POWER2(OC)) {
				if (!(OC & 15)) ksIms_u88R8_oc2pow(stream, 4, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, LOG2(OC), sh, sw, ph, pw, GN, GM);
				else ksIms_88R8_oc2pow(stream, 4, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, LOG2(OC), sh, sw, ph, pw, GN, GM);
			}
			else ksIms_88R8(stream, 4, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, OC, sh, sw, ph, pw, GN, GM);
			dconv3d_dX_ksImsR_Branch(127, 127); return;
		}
		if ((GN > 63) && (GM > 63)) {
			if (IS_POWER2(OC)) {
				if (!(OC & 7)) ksIms_u88R8_oc2pow(stream, 3, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, LOG2(OC), sh, sw, ph, pw, GN, GM);
				else ksIms_88R8_oc2pow(stream, 3, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, LOG2(OC), sh, sw, ph, pw, GN, GM);
			}
			else ksIms_88R8(stream, 3, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, OC, sh, sw, ph, pw, GN, GM);
			dconv3d_dX_ksImsR_Branch(63, 63); return;
		}
	}

	if (!(IH_slice & 3) && !(IW_slice & 3)) {//(IH_slice, IW_slice) % 4 == 0
		if ((GN > 127) && (GM > 127) && !(OC & 7)) {//OC % 8 == 0
			if (IS_POWER2(OC)) {
				if (!(OC & 15)) ksIms_u88R4_oc2pow(stream, 4, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, LOG2(OC), sh, sw, ph, pw, GN, GM);
				else ksIms_88R4_oc2pow(stream, 4, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, LOG2(OC), sh, sw, ph, pw, GN, GM);
			}
			else ksIms_88R4(stream, 4, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, OC, sh, sw, ph, pw, GN, GM);
			dconv3d_dX_ksImsR_Branch(127, 127); return;
		}
		if ((GN > 63) && (GM > 63)) {
			if (IS_POWER2(OC)) {
				if (!(OC & 7)) ksIms_u88R4_oc2pow(stream, 3, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, LOG2(OC), sh, sw, ph, pw, GN, GM);
				else ksIms_88R4_oc2pow(stream, 3, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, LOG2(OC), sh, sw, ph, pw, GN, GM);
			}
			else ksIms_88R4(stream, 3, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, OC, sh, sw, ph, pw, GN, GM);
			dconv3d_dX_ksImsR_Branch(63, 63); return;
		}
	}

	if ((GN > 127) && (GM > 127) && !(OC & 7)) {//OC % 8 == 0
		if (IS_POWER2(OC)) ksIms_88R_oc2pow(stream, 4, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, LOG2(OC), sh, sw, ph, pw, GN, GM);
		else ksIms_88R(stream, 4, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, OC, sh, sw, ph, pw, GN, GM);
		dconv3d_dX_ksImsR_Branch(127, 127); return;
	}
	if ((GN > 63) && (GM > 63)) {
		if (IS_POWER2(OC)) ksIms_88R_oc2pow(stream, 3, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, LOG2(OC), sh, sw, ph, pw, GN, GM);
		else ksIms_88R(stream, 3, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, OC, sh, sw, ph, pw, GN, GM);
		dconv3d_dX_ksImsR_Branch(63, 63); return;
	}

	if ((GN > 63) && (GM > 31)) {
		if (IS_POWER2(OC)) ksIms_84R_oc2pow(stream, 3, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, LOG2(OC), sh, sw, ph, pw, GN, GM);
		else ksIms_84R(stream, 3, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, OC, sh, sw, ph, pw, GN, GM);
		dconv3d_dX_ksImsR_Branch(63, 31); return;
	}

	if ((GN > 31) && (GM > 63)) {
		if (IS_POWER2(OC)) ksIms_48R_oc2pow(stream, 3, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, LOG2(OC), sh, sw, ph, pw, GN, GM);
		else ksIms_48R(stream, 3, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, OC, sh, sw, ph, pw, GN, GM);
		dconv3d_dX_ksImsR_Branch(31, 63); return;
	}

	if ((GN > 31) && (GM > 31)) {
		if (IS_POWER2(OC)) ksIms_44R_oc2pow(stream, 3, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, LOG2(OC), sh, sw, ph, pw, GN, GM);
		else ksIms_44R(stream, 3, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, OC, sh, sw, ph, pw, GN, GM);
		dconv3d_dX_ksImsR_Branch(31, 31); return;
	}

	if ((GN > 31) && (GM > 15)) {
		ksIms_42R(stream, 3, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, OC, sh, sw, ph, pw, GN, GM);
		dconv3d_dX_ksImsR_Branch(31, 15); return;
	}
	if ((GN > 15) && (GM > 31)) {
		ksIms_24R(stream, 3, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, OC, sh, sw, ph, pw, GN, GM);
		dconv3d_dX_ksImsR_Branch(15, 31); return;
	}

	if (OC > 7) {//BLOCK_SIZE = 8, OC >= 8
		if ((GN > 15) && (GM > 15)) {
			ksIms_22R(stream, 3, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, OC, sh, sw, ph, pw, GN, GM);
			dconv3d_dX_ksImsR_Branch(15, 15); return;
		}
		if ((GN > 15) && (GM > 7)) {
			ksIms_21R(stream, 3, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, OC, sh, sw, ph, pw, GN, GM);
			dconv3d_dX_ksImsR_Branch(15, 7); return;
		}
		if ((GN > 7) && (GM > 15)) {
			ksIms_12R(stream, 3, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, OC, sh, sw, ph, pw, GN, GM);
			dconv3d_dX_ksImsR_Branch(7, 15); return;
		}
		if ((GN > 7) && (GM > 7)) {
			ksIms_11R(stream, 3, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, OC, sh, sw, ph, pw, GN, GM);
			dconv3d_dX_ksImsR_Branch(7, 7); return;
		}
	}

	if ((GN > 7) && (GM > 7)) {
		ksIms_22R(stream, 2, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, OC, sh, sw, ph, pw, GN, GM);
		dconv3d_dX_ksImsR_Branch(7, 7); return;
	}
	if (GN > 7) {
		ksIms_21R(stream, 2, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, OC, sh, sw, ph, pw, GN, GM);
		dconv3d_dX_ksImsR_Branch(7, 3); return;
	}
	if (GM > 7) {
		ksIms_12R(stream, 2, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, OC, sh, sw, ph, pw, GN, GM);
		dconv3d_dX_ksImsR_Branch(3, 7); return;
	}
	ksIms_11R(stream, 2, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, OC, sh, sw, ph, pw, GN, GM);
}

#endif


//Sparse Kernel: for (IH, IW) % (sh, sw) = 0
#ifndef DCONV_3D_DELTAX_KERNEL_SPLIT_IMSR_TEXTURE
#define DCONV_3D_DELTAX_KERNEL_SPLIT_IMSR_TEXTURE

#ifndef DCONV_3D_DELTAX_KERNEL_SPLIT_IMSR_TEXTURE_MICRO
#define DCONV_3D_DELTAX_KERNEL_SPLIT_IMSR_TEXTURE_MICRO

#define __dconv3D_deltaX_ksImsR_tex(streams,index,length, texDy,deltaY,OH,OW, CW,FH,FW,CWstride, deltaX,IH_slice,IW_slice, IC,OC, sh, sw, ph, pw) \
	dconv3D_deltaX_ksImsR_texture(streams,index,length,\
		texDy,deltaY,OH,OW, CW,FH,FW,CWstride, deltaX,IH_slice,IW_slice, IC,OC, sh,sw,ph,pw,\
		GN,GM, 0,0)\

#define dconv3d_dX_ksImsR_Branch_tex(SIZE_Y, SIZE_X) {\
	int GNr = GN & (SIZE_Y), GMr = GM & (SIZE_X);\
	if(GNr && GMr){\
		int next_ic_index = (GN - GNr) + ic_index;\
		int next_j_index = (GM - GMr) + j_index;\
		dconv3D_deltaX_ksImsR_texture(streams,index,length, texDy,deltaY,OH,OW, CW,FH,FW,CWstride, deltaX,IH_slice,IW_slice, IC,OC, sh,sw,ph,pw,\
			GNr, GM,  next_ic_index, j_index);\
		dconv3D_deltaX_ksImsR_texture(streams,index,length, texDy,deltaY,OH,OW, CW,FH,FW,CWstride, deltaX,IH_slice,IW_slice, IC,OC, sh,sw,ph,pw,\
			GN,  GMr, ic_index, next_j_index);\
		dconv3D_deltaX_ksImsR_texture(streams,index,length, texDy,deltaY,OH,OW, CW,FH,FW,CWstride, deltaX,IH_slice,IW_slice, IC,OC, sh,sw,ph,pw,\
			GNr, GMr, next_ic_index, next_j_index);}\
	else if(GNr){\
		int next_ic_index = (GN - GNr) + ic_index;\
		dconv3D_deltaX_ksImsR_texture(streams,index,length, texDy,deltaY,OH,OW, CW,FH,FW,CWstride, deltaX,IH_slice,IW_slice, IC,OC, sh,sw,ph,pw,\
			GNr, GM, next_ic_index, j_index);}\
	else if(GMr){\
		int next_j_index = (GM - GMr) + j_index;\
		dconv3D_deltaX_ksImsR_texture(streams,index,length, texDy,deltaY,OH,OW, CW,FH,FW,CWstride, deltaX,IH_slice,IW_slice, IC,OC, sh,sw,ph,pw,\
			GN, GMr, ic_index, next_j_index);}}

#endif

//<1> IMS: INPUT_MOD_STEP: (IH, IW) % (sh, sw) == 0
//<2> remode: W[OC, FH, FW, IC] -> CW[sh, sw, OC, CFH, CFW, IC]
//<3> OC % 4 == 0, OC >= 4
//<4> GM = N * IH_slice * IW_slice, GM % 4 == 0, GM >= 4
//<5> GN = IC, GN % 4 == 0, GN >= 4
void dconv3D_deltaX_ksImsR_texture(jlong *streams, int &index, int length,
	cudaTextureObject_t texDy, const float* deltaY, int OH, int OW,
	const float*           CW, int FH, int FW, int CWstride,
	      float*       deltaX, int IH_slice, int IW_slice,
	int IC, int OC,
	int sh, int sw, int ph, int pw,
	int GN, int GM,
	int ic_index, int j_index)
{
	next_cudaStream(stream, streams, index, length);

	if (!(IH_slice & 7) && !(IW_slice & 7)) {//(IH_slice, IW_slice) % 8 == 0
		if ((GN > 127) && (GM > 127) && !(OC & 7)) {//OC % 8 == 0
			if (IS_POWER2(OC)) {
				if (!(OC & 15)) ksIms_u88R8_oc2pow(stream, 4, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, LOG2(OC), sh, sw, ph, pw, GN, GM);
				else ksIms_88R8_oc2pow(stream, 4, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, LOG2(OC), sh, sw, ph, pw, GN, GM);
			}
			else ksIms_88R8(stream, 4, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, OC, sh, sw, ph, pw, GN, GM);
			dconv3d_dX_ksImsR_Branch_tex(127, 127); return;
		}
		if ((GN > 63) && (GM > 63)) {
			if (IS_POWER2(OC)) {
				if (!(OC & 7)) ksIms_u88R8_oc2pow(stream, 3, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, LOG2(OC), sh, sw, ph, pw, GN, GM);
				else ksIms_88R8_oc2pow(stream, 3, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, LOG2(OC), sh, sw, ph, pw, GN, GM);
			}
			else ksIms_88R8_tex(stream, 3, ic_index, j_index, texDy, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, OC, sh, sw, ph, pw, GN, GM);
			dconv3d_dX_ksImsR_Branch_tex(63, 63); return;
		}
	}

	if (!(IH_slice & 3) && !(IW_slice & 3)) {//(IH_slice, IW_slice) % 4 == 0
		if ((GN > 127) && (GM > 127) && !(OC & 7)) {//OC % 8 == 0
			if (IS_POWER2(OC)) {
				if (!(OC & 15)) ksIms_u88R4_oc2pow(stream, 4, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, LOG2(OC), sh, sw, ph, pw, GN, GM);
				else ksIms_88R4_oc2pow_tex(stream, 4, ic_index, j_index, texDy, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, LOG2(OC), sh, sw, ph, pw, GN, GM);
			}
			else ksIms_88R4_tex(stream, 4, ic_index, j_index, texDy, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, OC, sh, sw, ph, pw, GN, GM);
			dconv3d_dX_ksImsR_Branch_tex(127, 127); return;
		}
		if ((GN > 63) && (GM > 63)) {
			if (IS_POWER2(OC)) {
				if (!(OC & 7)) ksIms_u88R4_oc2pow(stream, 3, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, LOG2(OC), sh, sw, ph, pw, GN, GM);
				else ksIms_88R4_oc2pow_tex(stream, 3, ic_index, j_index, texDy, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, LOG2(OC), sh, sw, ph, pw, GN, GM);
			}
			else ksIms_88R4_tex(stream, 3, ic_index, j_index, texDy, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, OC, sh, sw, ph, pw, GN, GM);
			dconv3d_dX_ksImsR_Branch_tex(63, 63); return;
		}
	}

	if ((GN > 127) && (GM > 127) && !(OC & 7)) {//OC % 8 == 0
		if (IS_POWER2(OC)) ksIms_88R_oc2pow_tex(stream, 4, ic_index, j_index, texDy, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, LOG2(OC), sh, sw, ph, pw, GN, GM);
		else ksIms_88R_tex(stream, 4, ic_index, j_index, texDy, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, OC, sh, sw, ph, pw, GN, GM);
		dconv3d_dX_ksImsR_Branch_tex(127, 127); return;
	}
	if ((GN > 63) && (GM > 63)) {
		if (IS_POWER2(OC)) ksIms_88R_oc2pow_tex(stream, 3, ic_index, j_index, texDy, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, LOG2(OC), sh, sw, ph, pw, GN, GM);
		else ksIms_88R_tex(stream, 3, ic_index, j_index, texDy, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, OC, sh, sw, ph, pw, GN, GM);
		dconv3d_dX_ksImsR_Branch_tex(63, 63); return;
	}

	if ((GN > 63) && (GM > 31)) {
		if (IS_POWER2(OC)) ksIms_84R_oc2pow_tex(stream, 3, ic_index, j_index, texDy, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, LOG2(OC), sh, sw, ph, pw, GN, GM);
		else ksIms_84R_tex(stream, 3, ic_index, j_index, texDy, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, OC, sh, sw, ph, pw, GN, GM);
		dconv3d_dX_ksImsR_Branch_tex(63, 31); return;
	}

	if ((GN > 31) && (GM > 63)) {
		if (IS_POWER2(OC)) ksIms_48R_oc2pow_tex(stream, 3, ic_index, j_index, texDy, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, LOG2(OC), sh, sw, ph, pw, GN, GM);
		else ksIms_48R_tex(stream, 3, ic_index, j_index, texDy, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, OC, sh, sw, ph, pw, GN, GM);
		dconv3d_dX_ksImsR_Branch_tex(31, 63); return;
	}

	if ((GN > 31) && (GM > 31)) {
		if (IS_POWER2(OC)) ksIms_44R_oc2pow_tex(stream, 3, ic_index, j_index, texDy, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, LOG2(OC), sh, sw, ph, pw, GN, GM);
		else ksIms_44R_tex(stream, 3, ic_index, j_index, texDy, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, OC, sh, sw, ph, pw, GN, GM);
		dconv3d_dX_ksImsR_Branch(31, 31); return;
	}

	if ((GN > 31) && (GM > 15)) {
		ksIms_42R(stream, 3, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, OC, sh, sw, ph, pw, GN, GM);
		dconv3d_dX_ksImsR_Branch(31, 15); return;
	}
	if ((GN > 15) && (GM > 31)) {
		ksIms_24R(stream, 3, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, OC, sh, sw, ph, pw, GN, GM);
		dconv3d_dX_ksImsR_Branch(15, 31); return;
	}

	if (OC > 7) {//BLOCK_SIZE = 8, OC >= 8
		if ((GN > 15) && (GM > 15)) {
			ksIms_22R(stream, 3, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, OC, sh, sw, ph, pw, GN, GM);
			dconv3d_dX_ksImsR_Branch(15, 15); return;
		}
		if ((GN > 15) && (GM > 7)) {
			ksIms_21R(stream, 3, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, OC, sh, sw, ph, pw, GN, GM);
			dconv3d_dX_ksImsR_Branch(15, 7); return;
		}
		if ((GN > 7) && (GM > 15)) {
			ksIms_12R(stream, 3, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, OC, sh, sw, ph, pw, GN, GM);
			dconv3d_dX_ksImsR_Branch(7, 15); return;
		}
		if ((GN > 7) && (GM > 7)) {
			ksIms_11R(stream, 3, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, OC, sh, sw, ph, pw, GN, GM);
			dconv3d_dX_ksImsR_Branch(7, 7); return;
		}
	}

	if ((GN > 7) && (GM > 7)) {
		ksIms_22R(stream, 2, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, OC, sh, sw, ph, pw, GN, GM);
		dconv3d_dX_ksImsR_Branch(7, 7); return;
	}
	if (GN > 7) {
		ksIms_21R(stream, 2, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, OC, sh, sw, ph, pw, GN, GM);
		dconv3d_dX_ksImsR_Branch(7, 3); return;
	}
	if (GM > 7) {
		ksIms_12R(stream, 2, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, OC, sh, sw, ph, pw, GN, GM);
		dconv3d_dX_ksImsR_Branch(3, 7); return;
	}
	ksIms_11R(stream, 2, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, OC, sh, sw, ph, pw, GN, GM);
}

#endif


//Sparse Kernel: for (IH, IW) % (sh, sw) = 0,  sh = sw = 2
#ifndef DCONV_3D_DELTAX_KERNEL_SPLIT_IMS2R
#define DCONV_3D_DELTAX_KERNEL_SPLIT_IMS2R

#ifndef DCONV_3D_DELTAX_KERNEL_SPLIT_IMS2R_MICRO
#define DCONV_3D_DELTAX_KERNEL_SPLIT_IMS2R_MICRO

#define __dconv3D_deltaX_ksIms2R(streams,index,length, deltaY,OH,OW, CW,FH,FW,CWstride, deltaX,IH_slice,IW_slice, N,IC,OC, ph, pw) \
	dconv3D_deltaX_ksIms2R(streams,index,length,\
		deltaY,OH,OW, CW,FH,FW,CWstride, deltaX,IH_slice,IW_slice, N,IC,OC, ph,pw,\
		GN,GM, 0,0)\

#define dconv3d_dX_ksIms2R_Branch(SIZE_Y, SIZE_X) {\
	int GNr = GN & (SIZE_Y), GMr = GM & (SIZE_X);\
	if(GNr && GMr){\
		int next_ic_index = (GN - GNr) + ic_index;\
		int next_j_index = (GM - GMr) + j_index;\
		dconv3D_deltaX_ksIms2R(streams,index,length, deltaY,OH,OW, CW,FH,FW,CWstride, deltaX,IH_slice,IW_slice, N,IC,OC, ph,pw,\
			GNr, GM,  next_ic_index, j_index);\
		dconv3D_deltaX_ksIms2R(streams,index,length, deltaY,OH,OW, CW,FH,FW,CWstride, deltaX,IH_slice,IW_slice, N,IC,OC, ph,pw,\
			GN,  GMr, ic_index, next_j_index);\
		dconv3D_deltaX_ksIms2R(streams,index,length, deltaY,OH,OW, CW,FH,FW,CWstride, deltaX,IH_slice,IW_slice, N,IC,OC, ph,pw,\
			GNr, GMr, next_ic_index, next_j_index);}\
	else if(GNr){\
		int next_ic_index = (GN - GNr) + ic_index;\
		dconv3D_deltaX_ksIms2R(streams,index,length, deltaY,OH,OW, CW,FH,FW,CWstride, deltaX,IH_slice,IW_slice, N,IC,OC, ph,pw,\
			GNr, GM, next_ic_index, j_index);}\
	else if(GMr){\
		int next_j_index = (GM - GMr) + j_index;\
		dconv3D_deltaX_ksIms2R(streams,index,length, deltaY,OH,OW, CW,FH,FW,CWstride, deltaX,IH_slice,IW_slice, N,IC,OC, ph,pw,\
			GN, GMr, ic_index, next_j_index);}}

#endif

//<1> IMS: INPUT_MOD_STEP:(IH, IW) % (sh, sw) == 0
//<2> sh = sw = 2
//<3> remode: W[OC, FH, FW, IC] -> CW[sh, sw, OC, CFH, CFW, IC]
//<4> OC % 4 == 0, OC >= 4
//<5> GM = N * IH_slice * IW_slice, GM % 4 == 0, GM >= 4
//<6> GN = IC, GN % 4 == 0, GN >= 4
void dconv3D_deltaX_ksIms2R(jlong *streams, int &index, int length,
	const float* deltaY, int OH, int OW,
	const float*     CW, int FH, int FW, int CWstride,
	      float* deltaX, int IH_slice, int IW_slice,
	int N, int IC, int OC,
	int ph, int pw,
	int GN, int GM,
	int ic_index, int j_index)
{
	next_cudaStream(stream, streams, index, length);

	if ((GN > 127) && (GM > 127) && !(OC & 7)) {//[128, 128], OC % 8 == 0
		//======[OC && CFW is power of 2]===================================================
		if (IS_POWER2(OC) && Ims2_IS_CW_POWER2(FW)) {
			if (!(IH_slice & 7) && !(IW_slice & 7)) {
				if (!(OC & 15)) {
					if(OC > 127) ksIms2_u88R8_oc_CFW2pow(stream, 4, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, LOG2(OC), ph, pw, GN, GM);
					else ksIms2_u88R8_oc_CFW2pow(stream, 3, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, LOG2(OC), ph, pw, GN, GM);
				}
				else ksIms2_88R8_oc_CFW2pow(stream, 4, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, LOG2(OC), ph, pw, GN, GM);
			}
			else if (!(IH_slice & 3) && !(IW_slice & 3)) {
				if(!(OC & 15)) ksIms2_u88R4_oc_CFW2pow(stream, 4, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, LOG2(OC), ph, pw, GN, GM);
				else ksIms2_88R4_oc_CFW2pow(stream, 4, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, LOG2(OC), ph, pw, GN, GM);
			}
			else ksIms2_88R_oc_CFW2pow(stream, 4, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, LOG2(OC), ph, pw, GN, GM);
		}
		//======[CFH, CFW is power of 2]====================================================
		else if (Ims2_IS_CW_POWER2(FH) && Ims2_IS_CW_POWER2(FW)) {
			if (!(IH_slice & 7) && !(IW_slice & 7)) ksIms2_88R8_CW2pow(stream, 4, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, GM);
			else if (!(IH_slice & 3) && !(IW_slice & 3)) ksIms2_88R4_CW2pow(stream, 4, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, GM);
			else if (!(N & 7)) ksIms2A_88R_CW2pow(stream, 4, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, N, IC, OC, ph, pw, GN, GM);
			else ksIms2_88R_CW2pow(stream, 4, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, GM);
		}
		//======[OC is power of 2]==========================================================
		else if (IS_POWER2(OC)) {
			if (!(IH_slice & 7) && !(IW_slice & 7)) ksIms2_88R8_oc2pow(stream, 4, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, LOG2(OC), ph, pw, GN, GM);
			else if (!(IH_slice & 3) && !(IW_slice & 3)) ksIms2_88R4_oc2pow(stream, 4, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, LOG2(OC), ph, pw, GN, GM);
			else ksIms2_88R_oc2pow(stream, 4, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, LOG2(OC), ph, pw, GN, GM);
		}
		//======[Common]====================================================================
		else if (!(N & 7)) ksIms2A_88R(stream, 4, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, N, IC, OC, ph, pw, GN, GM);
		else if (!(IH_slice & 7) && !(IW_slice & 7)) ksIms2_88R8(stream, 4, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, GM);
		else if (!(IH_slice & 3) && !(IW_slice & 3)) ksIms2_88R4(stream, 4, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, GM);
		else ksIms2_88R(stream, 4, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, GM);
		dconv3d_dX_ksIms2R_Branch(127, 127); return;
	}

	if ((GN > 63) && (GM > 63)) {//[64, 64]
		//======[OC && CFW is power of 2]===================================================
		if (IS_POWER2(OC) && Ims2_IS_CW_POWER2(FW)) {
			if (!(IH_slice & 7) && !(IW_slice & 7)) {
				if (!(OC & 7)) ksIms2_u88R8_oc_CFW2pow(stream, 3, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, LOG2(OC), ph, pw, GN, GM);
				else ksIms2_88R8_oc_CFW2pow(stream, 3, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, LOG2(OC), ph, pw, GN, GM);
			}
			else if (!(IH_slice & 3) && !(IW_slice & 3)) {
				if (!(OC & 7)) ksIms2_u88R4_oc_CFW2pow(stream, 3, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, LOG2(OC), ph, pw, GN, GM);
				else ksIms2_88R4_oc_CFW2pow(stream, 3, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, LOG2(OC), ph, pw, GN, GM);
			}
			else if (!(OC & 7)) ksIms2_u88R_oc_CFW2pow(stream, 3, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, LOG2(OC), ph, pw, GN, GM);
			else ksIms2_88R_oc_CFW2pow(stream, 3, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, LOG2(OC), ph, pw, GN, GM);
		}
		//======[CFH, CFW is power of 2]====================================================
		else if (Ims2_IS_CW_POWER2(FH) && Ims2_IS_CW_POWER2(FW)) {
			if (!(IH_slice & 7) && !(IW_slice & 7)) ksIms2_88R8_CW2pow(stream, 3, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, GM);
			else if (!(IH_slice & 3) && !(IW_slice & 3)) ksIms2_88R4_CW2pow(stream, 3, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, GM);
			else ksIms2_88R_CW2pow(stream, 3, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, GM);
		}
		//======[OC is power of 2]==========================================================
		else if (IS_POWER2(OC)) {
			if (!(IH_slice & 7) && !(IW_slice & 7)) ksIms2_88R8_oc2pow(stream, 3, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, LOG2(OC), ph, pw, GN, GM);
			else if (!(IH_slice & 3) && !(IW_slice & 3)) ksIms2_88R4_oc2pow(stream, 3, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, LOG2(OC), ph, pw, GN, GM);
			else ksIms2_88R_oc2pow(stream, 3, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, LOG2(OC), ph, pw, GN, GM);
		}
		//======[Common]====================================================================
		else if (!(N & 7)) ksIms2A_88R(stream, 3, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, N, IC, OC, ph, pw, GN, GM);
		else if (!(IH_slice & 7) && !(IW_slice & 7)) ksIms2_88R8(stream, 3, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, GM);
		else if (!(IH_slice & 3) && !(IW_slice & 3)) ksIms2_88R4(stream, 3, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, GM);
		else ksIms2_88R(stream, 3, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, GM);
		dconv3d_dX_ksIms2R_Branch(63, 63); return;
	}

	if ((GN > 63) && (GM > 31)) {
		if (IS_POWER2(OC)) ksIms2_84R_oc2pow(stream, 3, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, LOG2(OC), ph, pw, GN, GM);
		else ksIms2_84R(stream, 3, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, GM);
		dconv3d_dX_ksIms2R_Branch(63, 31); return;
	}

	if ((GN > 31) && (GM > 63)) {
		if (IS_POWER2(OC)) ksIms2_48R_oc2pow(stream, 3, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, LOG2(OC), ph, pw, GN, GM);
		else ksIms2_48R(stream, 3, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, GM);
		dconv3d_dX_ksIms2R_Branch(31, 63); return;
	}

	if ((GN > 31) && (GM > 31)) {
		if (IS_POWER2(OC)) ksIms2_44R_oc2pow(stream, 3, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, LOG2(OC), ph, pw, GN, GM);
		else ksIms2_44R(stream, 3, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, GM);
		dconv3d_dX_ksIms2R_Branch(31, 31); return;
	}

	if ((GN > 31) && (GM > 15)) {
		ksIms2_42R(stream, 3, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, GM);
		dconv3d_dX_ksIms2R_Branch(31, 15); return;
	}
	if ((GN > 15) && (GM > 31)) {
		ksIms2_24R(stream, 3, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, GM);
		dconv3d_dX_ksIms2R_Branch(15, 31); return;
	}

	if (OC > 7) {//BLOCK_SIZE = 8, OC >= 8
		if ((GN > 15) && (GM > 15)) {
			ksIms_22R(stream, 3, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, OC, 2, 2, ph, pw, GN, GM);
			dconv3d_dX_ksIms2R_Branch(15, 15); return;
		}
		if ((GN > 15) && (GM > 7)) {
			ksIms_21R(stream, 3, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, OC, 2, 2, ph, pw, GN, GM);
			dconv3d_dX_ksIms2R_Branch(15, 7); return;
		}
		if ((GN > 7) && (GM > 15)) {
			ksIms_12R(stream, 3, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, OC, 2, 2, ph, pw, GN, GM);
			dconv3d_dX_ksIms2R_Branch(7, 15); return;
		}
		if ((GN > 7) && (GM > 7)) {
			ksIms_11R(stream, 3, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, OC, 2, 2, ph, pw, GN, GM);
			dconv3d_dX_ksIms2R_Branch(7, 7); return;
		}
	}

	if ((GN > 7) && (GM > 7)) {
		ksIms_22R(stream, 2, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, OC, 2, 2, ph, pw, GN, GM);
		dconv3d_dX_ksIms2R_Branch(7, 7); return;
	}
	if (GN > 7) {
		ksIms_21R(stream, 2, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, OC, 2, 2, ph, pw, GN, GM);
		dconv3d_dX_ksIms2R_Branch(7, 3); return;
	}
	if (GM > 7) {
		ksIms_12R(stream, 2, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, OC, 2, 2, ph, pw, GN, GM);
		dconv3d_dX_ksIms2R_Branch(3, 7); return;
	}
	ksIms_11R(stream, 2, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, OC, 2, 2, ph, pw, GN, GM);
}

#endif


//Sparse Kernel : for (IH, IW) % (sh, sw) = 0, sh = sw = 2
#ifndef DCONV_3D_DELTAX_KERNEL_SPLIT_IMS2R_TEXTURE
#define DCONV_3D_DELTAX_KERNEL_SPLIT_IMS2R_TEXTURE

#ifndef DCONV_3D_DELTAX_KERNEL_SPLIT_IMS2R_TEXTURE_MICRO
#define DCONV_3D_DELTAX_KERNEL_SPLIT_IMS2R_TEXTURE_MICRO

#define __dconv3D_deltaX_ksIms2R_tex(streams,index,length, texDy,deltaY,OH,OW, CW,FH,FW,CWstride, deltaX,IH_slice,IW_slice, IC,OC, ph, pw) \
	dconv3D_deltaX_ksIms2R_texture(streams,index,length,\
		texDy,deltaY,OH,OW, CW,FH,FW,CWstride, deltaX,IH_slice,IW_slice, N,IC,OC, ph,pw,\
		GN, GM, 0,0)\

#define dconv3d_dX_ksIms2R_Branch_tex(SIZE_Y, SIZE_X) {\
	int GNr = GN & (SIZE_Y), GMr = GM & (SIZE_X);\
	if(GNr && GMr){\
		int next_ic_index = (GN - GNr) + ic_index;\
		int next_j_index = (GM - GMr) + j_index;\
		dconv3D_deltaX_ksIms2R_texture(streams,index,length, texDy,deltaY,OH,OW, CW,FH,FW,CWstride, deltaX,IH_slice,IW_slice, N,IC,OC, ph,pw,\
			GNr, GM,  next_ic_index, j_index);\
		dconv3D_deltaX_ksIms2R_texture(streams,index,length, texDy,deltaY,OH,OW, CW,FH,FW,CWstride, deltaX,IH_slice,IW_slice, N,IC,OC, ph,pw,\
			GN,  GMr, ic_index, next_j_index);\
		dconv3D_deltaX_ksIms2R_texture(streams,index,length, texDy,deltaY,OH,OW, CW,FH,FW,CWstride, deltaX,IH_slice,IW_slice, N,IC,OC, ph,pw,\
			GNr, GMr, next_ic_index, next_j_index);}\
	else if(GNr){\
		int next_ic_index = (GN - GNr) + ic_index;\
		dconv3D_deltaX_ksIms2R_texture(streams,index,length, texDy,deltaY,OH,OW, CW,FH,FW,CWstride, deltaX,IH_slice,IW_slice, N,IC,OC, ph,pw,\
			GNr, GM, next_ic_index, j_index);}\
	else if(GMr){\
		int next_j_index = (GM - GMr) + j_index;\
		dconv3D_deltaX_ksIms2R_texture(streams,index,length, texDy,deltaY,OH,OW, CW,FH,FW,CWstride, deltaX,IH_slice,IW_slice, N,IC,OC, ph,pw,\
			GN, GMr, ic_index, next_j_index);}}

#endif

//<1> IMS: INPUT_MOD_STEP:(IH, IW) % (sh, sw) == 0
//<2> sh = sw = 2
//<3> remode: W[OC, FH, FW, IC] -> CW[sh, sw, OC, CFH, CFW, IC]
//<4> OC % 4 == 0, OC >= 4
//<5> GM = N * IH_slice * IW_slice, GM % 4 == 0, GM >= 4
//<6> GN = IC, GN % 4 == 0, GN >= 4
void dconv3D_deltaX_ksIms2R_texture(jlong *streams, int &index, int length,
	cudaTextureObject_t texDy, const float* deltaY, int OH, int OW,
	const float*           CW, int FH, int FW, int CWstride,
	      float*       deltaX, int IH_slice, int IW_slice,
	int N, int IC, int OC,
	int ph, int pw,
	int GN, int GM,
	int ic_index, int j_index)
{
	next_cudaStream(stream, streams, index, length);

	if ((GN > 127) && (GM > 127) && !(OC & 7)) {//[128, 128], OC % 8 == 0
		//======[OC && CFW is power of 2]===================================================
		if (IS_POWER2(OC) && Ims2_IS_CW_POWER2(FW)) {
			if (!(IH_slice & 7) && !(IW_slice & 7)) {
				if (!(OC & 15)) {
					if (OC > 127) ksIms2_u88R8_oc_CFW2pow(stream, 4, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, LOG2(OC), ph, pw, GN, GM);
					else ksIms2_u88R8_oc_CFW2pow(stream, 3, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, LOG2(OC), ph, pw, GN, GM);
				}
				else ksIms2_88R8_oc_CFW2pow(stream, 4, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, LOG2(OC), ph, pw, GN, GM);
			}
			else if (!(IH_slice & 3) && !(IW_slice & 3)) {
				if (!(OC & 15)) ksIms2_u88R4_oc_CFW2pow(stream, 4, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, LOG2(OC), ph, pw, GN, GM);
				else ksIms2_88R4_oc_CFW2pow(stream, 4, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, LOG2(OC), ph, pw, GN, GM);
			}
			else ksIms2_88R_oc_CFW2pow(stream, 4, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, LOG2(OC), ph, pw, GN, GM);
		}
		//======[CFH, CFW is power of 2]====================================================
		else if (Ims2_IS_CW_POWER2(FH) && Ims2_IS_CW_POWER2(FW)) {
			if (!(IH_slice & 7) && !(IW_slice & 7)) ksIms2_88R8_CW2pow_tex(stream, 4, ic_index, j_index, texDy, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, GM);
			else if (!(IH_slice & 3) && !(IW_slice & 3)) ksIms2_88R4_CW2pow_tex(stream, 4, ic_index, j_index, texDy, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, GM);
			else if (!(N & 7)) ksIms2A_88R_CW2pow(stream, 4, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, N, IC, OC, ph, pw, GN, GM);
			else ksIms2_88R_CW2pow(stream, 4, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, GM);
		}
		//======[OC is power of 2]==========================================================
		else if (IS_POWER2(OC)) {
			if (!(IH_slice & 7) && !(IW_slice & 7)) ksIms2_88R8_oc2pow(stream, 4, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, LOG2(OC), ph, pw, GN, GM);
			else if (!(IH_slice & 3) && !(IW_slice & 3)) ksIms2_88R4_oc2pow(stream, 4, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, LOG2(OC), ph, pw, GN, GM);
			else ksIms2_88R_oc2pow(stream, 4, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, LOG2(OC), ph, pw, GN, GM);
		}
		//======[Common]====================================================================
		else if (!(N & 7)) ksIms2A_88R(stream, 4, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, N, IC, OC, ph, pw, GN, GM);
		else if (!(IH_slice & 7) && !(IW_slice & 7)) ksIms2_88R8_tex(stream, 4, ic_index, j_index, texDy, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, GM);
		else if (!(IH_slice & 3) && !(IW_slice & 3)) ksIms2_88R4_tex(stream, 4, ic_index, j_index, texDy, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, GM);
		else ksIms2_88R_tex(stream, 4, ic_index, j_index, texDy, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, GM);
		dconv3d_dX_ksIms2R_Branch_tex(127, 127); return;
	}

	if ((GN > 63) && (GM > 63)) {//[64, 64]
		//======[OC && CFW is power of 2]===================================================
		if (IS_POWER2(OC) && Ims2_IS_CW_POWER2(FW)) {
			if (!(IH_slice & 7) && !(IW_slice & 7)) {
				if (!(OC & 7)) ksIms2_u88R8_oc_CFW2pow(stream, 3, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, LOG2(OC), ph, pw, GN, GM);
				else ksIms2_88R8_oc_CFW2pow_tex(stream, 3, ic_index, j_index, texDy, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, LOG2(OC), ph, pw, GN, GM);
			}
			else if (!(IH_slice & 3) && !(IW_slice & 3)) {
				if (!(OC & 7)) ksIms2_u88R4_oc_CFW2pow(stream, 3, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, LOG2(OC), ph, pw, GN, GM);
				else ksIms2_88R4_oc_CFW2pow_tex(stream, 3, ic_index, j_index, texDy, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, LOG2(OC), ph, pw, GN, GM);
			}
			else if (!(OC & 7)) ksIms2_u88R_oc_CFW2pow(stream, 3, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, LOG2(OC), ph, pw, GN, GM);
			else ksIms2_88R_oc_CFW2pow(stream, 3, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, LOG2(OC), ph, pw, GN, GM);
		}
		//======[CFH, CFW is power of 2]====================================================
		else if (Ims2_IS_CW_POWER2(FH) && Ims2_IS_CW_POWER2(FW)) {
			if (!(IH_slice & 7) && !(IW_slice & 7)) ksIms2_88R8_CW2pow_tex(stream, 3, ic_index, j_index, texDy, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, GM);
			else if (!(IH_slice & 3) && !(IW_slice & 3)) ksIms2_88R4_CW2pow_tex(stream, 3, ic_index, j_index, texDy, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, GM);
			else ksIms2_88R_CW2pow(stream, 3, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, GM);
		}
		//======[OC is power of 2]==========================================================
		else if (IS_POWER2(OC)) {
			if (!(IH_slice & 7) && !(IW_slice & 7)) ksIms2_88R8_oc2pow_tex(stream, 3, ic_index, j_index, texDy, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, LOG2(OC), ph, pw, GN, GM);
			else if (!(IH_slice & 3) && !(IW_slice & 3)) ksIms2_88R4_oc2pow(stream, 3, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, LOG2(OC), ph, pw, GN, GM);
			else ksIms2_88R_oc2pow_tex(stream, 3, ic_index, j_index, texDy, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, LOG2(OC), ph, pw, GN, GM);
		}
		//======[Common]====================================================================
		else if (!(N & 7)) ksIms2A_88R(stream, 3, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, N, IC, OC, ph, pw, GN, GM);
		else if (!(IH_slice & 7) && !(IW_slice & 7)) ksIms2_88R8_tex(stream, 3, ic_index, j_index, texDy, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, GM);
		else if (!(IH_slice & 3) && !(IW_slice & 3)) ksIms2_88R4_tex(stream, 3, ic_index, j_index, texDy, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, GM);
		else ksIms2_88R_tex(stream, 3, ic_index, j_index, texDy, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, GM);
		dconv3d_dX_ksIms2R_Branch_tex(63, 63); return;
	}

	if ((GN > 63) && (GM > 31)) {
		if (IS_POWER2(OC)) ksIms2_84R_oc2pow_tex(stream, 3, ic_index, j_index, texDy, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, LOG2(OC), ph, pw, GN, GM);
		else ksIms2_84R_oc2pow_tex(stream, 3, ic_index, j_index, texDy, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, GM);
		dconv3d_dX_ksIms2R_Branch_tex(63, 31); return;
	}

	if ((GN > 31) && (GM > 63)) {
		if (IS_POWER2(OC)) ksIms2_48R_oc2pow_tex(stream, 3, ic_index, j_index, texDy, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, LOG2(OC), ph, pw, GN, GM);
		else ksIms2_48R_tex(stream, 3, ic_index, j_index, texDy, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, GM);
		dconv3d_dX_ksIms2R_Branch_tex(31, 63); return;
	}

	if ((GN > 31) && (GM > 31)) {
		if (IS_POWER2(OC)) ksIms2_44R_oc2pow_tex(stream, 3, ic_index, j_index, texDy, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, LOG2(OC), ph, pw, GN, GM);
		else ksIms2_44R_tex(stream, 3, ic_index, j_index, texDy, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, GM);
		dconv3d_dX_ksIms2R_Branch(31, 31); return;
	}

	if ((GN > 31) && (GM > 15)) {
		ksIms2_42R(stream, 3, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, GM);
		dconv3d_dX_ksIms2R_Branch(31, 15); return;
	}
	if ((GN > 15) && (GM > 31)) {
		ksIms2_24R(stream, 3, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, GM);
		dconv3d_dX_ksIms2R_Branch(15, 31); return;
	}

	if (OC > 7) {//BLOCK_SIZE = 8, OC >= 8
		if ((GN > 15) && (GM > 15)) {
			ksIms_22R(stream, 3, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, OC, 2, 2, ph, pw, GN, GM);
			dconv3d_dX_ksIms2R_Branch(15, 15); return;
		}
		if ((GN > 15) && (GM > 7)) {
			ksIms_21R(stream, 3, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, OC, 2, 2, ph, pw, GN, GM);
			dconv3d_dX_ksIms2R_Branch(15, 7); return;
		}
		if ((GN > 7) && (GM > 15)) {
			ksIms_12R(stream, 3, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, OC, 2, 2, ph, pw, GN, GM);
			dconv3d_dX_ksIms2R_Branch(7, 15); return;
		}
		if ((GN > 7) && (GM > 7)) {
			ksIms_11R(stream, 3, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, OC, 2, 2, ph, pw, GN, GM);
			dconv3d_dX_ksIms2R_Branch(7, 7); return;
		}
	}

	if ((GN > 7) && (GM > 7)) {
		ksIms_22R(stream, 2, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, OC, 2, 2, ph, pw, GN, GM);
		dconv3d_dX_ksIms2R_Branch(7, 7); return;
	}
	if (GN > 7) {
		ksIms_21R(stream, 2, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, OC, 2, 2, ph, pw, GN, GM);
		dconv3d_dX_ksIms2R_Branch(7, 3); return;
	}
	if (GM > 7) {
		ksIms_12R(stream, 2, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, OC, 2, 2, ph, pw, GN, GM);
		dconv3d_dX_ksIms2R_Branch(3, 7); return;
	}
	ksIms_11R(stream, 2, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, OC, 2, 2, ph, pw, GN, GM);
}

#endif


//Sparse Kernel: for (IH, IW) % (sh, sw) = 0,  sh = sw = 2
#ifndef DCONV_3D_DELTAX_KERNEL_SPLIT_V2_IMS2R
#define DCONV_3D_DELTAX_KERNEL_SPLIT_V2_IMS2R

#ifndef DCONV_3D_DELTAX_KERNEL_SPLIT_V2_IMS2R_MICRO
#define DCONV_3D_DELTAX_KERNEL_SPLIT_V2_IMS2R_MICRO

#define __dconv3D_deltaX_ksV2_Ims2R(env,streams,index,length, deltaY,OH,OW,sizeY, CW,FH,FW,CWstride, deltaX,IH_slice,IW_slice, IC,OC, ph,pw) \
	dconv3D_deltaX_ksV2_Ims2R(env,streams,index,length,\
		deltaY,OH,OW,sizeY, CW,FH,FW,CWstride, deltaX,IH_slice,IW_slice, IC,OC, ph,pw,\
		IC,N, 0,0)\

#define dconv3d_dX_ksV2_Ims2R_Branch(SIZE_Y, SIZE_X) {\
	int GNr = GN & (SIZE_Y), Nr = N & (SIZE_X);\
	if(GNr && Nr){\
		int next_ic_index = (GN - GNr) + ic_index;\
		int next_n_index = (N - Nr) + n_index;\
		dconv3D_deltaX_ksV2_Ims2R(env,streams,index,length, deltaY,OH,OW,sizeY, CW,FH,FW,CWstride, deltaX,IH_slice,IW_slice, IC,OC, ph,pw,\
			GNr, N,  next_ic_index, n_index);\
		dconv3D_deltaX_ksV2_Ims2R(env,streams,index,length, deltaY,OH,OW,sizeY, CW,FH,FW,CWstride, deltaX,IH_slice,IW_slice, IC,OC, ph,pw,\
			GN,  Nr, ic_index, next_n_index);\
		dconv3D_deltaX_ksV2_Ims2R(env,streams,index,length, deltaY,OH,OW,sizeY, CW,FH,FW,CWstride, deltaX,IH_slice,IW_slice, IC,OC, ph,pw,\
			GNr, Nr, next_ic_index, next_n_index);}\
	else if(GNr){\
		int next_ic_index = (GN - GNr) + ic_index;\
		dconv3D_deltaX_ksV2_Ims2R(env,streams,index,length, deltaY,OH,OW,sizeY, CW,FH,FW,CWstride, deltaX,IH_slice,IW_slice, IC,OC, ph,pw,\
			GNr, N, next_ic_index, n_index);}\
	else if(Nr){\
		int next_n_index = (N - Nr) + n_index;\
		dconv3D_deltaX_ksV2_Ims2R(env,streams,index,length, deltaY,OH,OW,sizeY, CW,FH,FW,CWstride, deltaX,IH_slice,IW_slice, IC,OC, ph,pw,\
			GN, Nr, ic_index, next_n_index);}}

#endif

//<1> IMS: INPUT_MOD_STEP: (IH, IW) % (sh, sw) == 0
//<2> sh = sw = 2
//<3> remode: W[OC, FH, FW, IC] -> CW[sh, sw, OC, CFH, CFW, IC]
//<4> OC % 4 == 0, OC >= 4
//<5> GM =  N, GM % 4 == 0, GM >= 4
//<6> GN = IC, GN % 4 == 0, GN >= 4
void dconv3D_deltaX_ksV2_Ims2R(JNIEnv *env, jlong *streams, int &index, int length,
	const float* deltaY, int OH, int OW, int sizeY,
	const float*     CW, int FH, int FW, int CWstride,
	      float* deltaX, int IH_slice, int IW_slice,
	int IC, int OC,
	int ph, int pw,
	int GN, int N,
	int ic_index, int n_index)
{
	next_cudaStream(stream, streams, index, length);

	if ((GN > 127) && (N > 127) && !(OC & 7)) {//[128, 128], OC % 8 == 0
		if (Ims2_IS_CW_POWER2(FW) && IS_POWER2(OC)) {//FH = 2, 3, 4, OC is power of 2
			if (!(OC & 15)) ksV2_Ims2_u88R_CFW_oc2pow(stream, 4, ic_index, n_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, LOG2(OC), ph, pw, GN, N);
			else ksV2_Ims2_88R_CFW_oc2pow(stream, 4, ic_index, n_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, LOG2(OC), ph, pw, GN, N);
		}
		else if (Ims2_CAN_W5(FH, FW, OH, OW) && IS_POWER2(OC)) {//FH = FW = 5, OC is power of 2
			if (!(OC & 15)) ksV2_Ims2_u88R_W5_oc2pow(stream, 4, ic_index, n_index, deltaY, OH, OW, CW, deltaX, IH_slice, IW_slice, IC, LOG2(OC), ph, pw, GN, N);
			else ksV2_Ims2_88R_W5_oc2pow(stream, 4, ic_index, n_index, deltaY, OH, OW, CW, deltaX, IH_slice, IW_slice, IC, LOG2(OC), ph, pw, GN, N);
		}
		else if (!(OC & 15)) ksV2_Ims2_u88R(stream, 4, ic_index, n_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, N);
		else ksV2_Ims2_88R(stream, 4, ic_index, n_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, N);
		dconv3d_dX_ksV2_Ims2R_Branch(127, 127); return;
	}

	if ((GN > 63) && (N > 63)) {//[64, 64], OC % 4 == 0
		if (Ims2_IS_CW_POWER2(FW) && IS_POWER2(OC)) {//FH = 2, 3, 4, OC is power of 2
			if (!(OC & 15)) ksV2_Ims2_u88R_CFW_oc2pow(stream, 3, ic_index, n_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, LOG2(OC), ph, pw, GN, N);
			else ksV2_Ims2_88R_CFW_oc2pow(stream, 3, ic_index, n_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, LOG2(OC), ph, pw, GN, N);
		}
		else if (Ims2_CAN_W5(FH, FW, OH, OW) && IS_POWER2(OC)) {//FH = FW = 5, OC is power of 2
			if (!(OC & 15)) ksV2_Ims2_u88R_W5_oc2pow(stream, 3, ic_index, n_index, deltaY, OH, OW, CW, deltaX, IH_slice, IW_slice, IC, LOG2(OC), ph, pw, GN, N);
			else ksV2_Ims2_88R_W5_oc2pow(stream, 3, ic_index, n_index, deltaY, OH, OW, CW, deltaX, IH_slice, IW_slice, IC, LOG2(OC), ph, pw, GN, N);
		}
		else if (!(OC & 15)) ksV2_Ims2_u88R(stream, 3, ic_index, n_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, N);
		else ksV2_Ims2_88R(stream, 3, ic_index, n_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, N);
		dconv3d_dX_ksV2_Ims2R_Branch(63, 63); return;
	}

	float Q; Ims2_PADDING_SCALE_UP(Q, IH_slice, IW_slice, OH, OW, FH, FW);
	if (Q <= 1.4) {
		KS_V2_TO_V1(FH, FW, IH_slice, IW_slice, N, OC, n_index);
		if (index > 0) index = index - 1;

		if (env) {//env != null, use texture
			cudaTextureObject_t texDy = floatTexture((float*)deltaY, sizeY, env);
			dconv3D_deltaX_ksIms2R_texture(streams, index, length,
				texDy, deltaY, OH, OW, CW, FH, FW, CWstride, deltaX, IH_slice, IW_slice, N, IC, OC, ph, pw,
				GN, GM, ic_index, j_index);
			cudaError_t error = cudaDestroyTextureObject(texDy); handleError(error);
			return;
		}

		dconv3D_deltaX_ksIms2R(streams, index, length,
			deltaY, OH, OW, CW, FH, FW, CWstride, deltaX, IH_slice, IW_slice, N, IC, OC, ph, pw,
			GN, GM, ic_index, j_index);
		return;
	}

	if ((GN > 63) && (N > 31)) {
		ksV2_Ims2_84R(stream, 3, ic_index, n_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, N);
		dconv3d_dX_ksV2_Ims2R_Branch(63, 31); return;
	}
	if ((GN > 31) && (N > 63)) {
		ksV2_Ims2_48R(stream, 3, ic_index, n_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, N);
		dconv3d_dX_ksV2_Ims2R_Branch(31, 63); return;
	}
	if ((GN > 31) && (N > 31)) {
		ksV2_Ims2_44R(stream, 3, ic_index, n_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, N);
		dconv3d_dX_ksV2_Ims2R_Branch(31, 31); return;
	}

	KS_V2_TO_V1(FH, FW, IH_slice, IW_slice, N, OC, n_index);
	if (index > 0) index = index - 1;

	if (env) {//env != null, use texture
		cudaTextureObject_t texDy = floatTexture((float*)deltaY, sizeY, env);
		dconv3D_deltaX_ksIms2R_texture(streams, index, length,
			texDy, deltaY, OH, OW, CW, FH, FW, CWstride, deltaX, IH_slice, IW_slice, N, IC, OC, ph, pw,
			GN, GM, ic_index, j_index);
		cudaError_t error = cudaDestroyTextureObject(texDy); handleError(error);
		return;
	}

	dconv3D_deltaX_ksIms2R(streams, index, length,
		deltaY, OH, OW, CW, FH, FW, CWstride, deltaX, IH_slice, IW_slice, N, IC, OC, ph, pw,
		GN, GM, ic_index, j_index);
}

#endif

#endif



//cross add: reversed forward porpagation
#ifndef CROSS_ADD_AREA
#define CROSS_ADD_AREA

#ifndef DECONV_3D_DELTAX_CROSS_ADD
#define DECONV_3D_DELTAX_CROSS_ADD

#ifndef DECONV_3D_DELTAX_CROSS_ADD_MICRO
#define DECONV_3D_DELTAX_CROSS_ADD_MICRO

#define __dconv3D_deltaX_CrossAdd(streams,index,length, deltaY,OH,OW, W,FH,FW, deltaX,IH,IW, N,IC,OC, sh,sw,ph,pw) \
	dconv3d_deltaX_CrossAdd(streams,index,length, deltaY,OH,OW, W,FH,FW, deltaX,IH,IW, IC,OC, sh,sw,ph,pw,\
		GET_GN_CrossAdd(OC), GET_GM_CrossAdd(N, OH, OW), GET_GK_CrossAdd(FH, FW, IC), 0, 0)\

#define dconv3d_dX_CrossAdd_Branch(SIZE_Y, SIZE_X) {\
	int GNr = GN & (SIZE_Y), GMr = GM & (SIZE_X);\
	if(GNr && GMr){\
		int next_oc_index = (GN - GNr) + oc_index;\
		int next_j_index = (GM - GMr) + j_index;\
		dconv3d_deltaX_CrossAdd(streams, index, length, deltaY,OH,OW, W,FH,FW, deltaX,IH,IW, IC,OC, sh,sw,ph,pw,\
			GNr, GM, GK, next_oc_index, j_index);\
		dconv3d_deltaX_CrossAdd(streams, index, length, deltaY,OH,OW, W,FH,FW, deltaX,IH,IW, IC,OC, sh,sw,ph,pw,\
			GN, GMr, GK, oc_index, next_j_index);\
		dconv3d_deltaX_CrossAdd(streams, index, length, deltaY,OH,OW, W,FH,FW, deltaX,IH,IW, IC,OC, sh,sw,ph,pw,\
			GNr, GMr, GK, next_oc_index, next_j_index);}\
	else if(GNr){\
		int next_oc_index = (GN - GNr) + oc_index;\
		dconv3d_deltaX_CrossAdd(streams, index, length, deltaY,OH,OW, W,FH,FW, deltaX,IH,IW, IC,OC, sh,sw,ph,pw,\
			GNr, GM, GK, next_oc_index, j_index);}\
	else if(GMr){\
		int next_j_index = (GM - GMr) + j_index;\
		dconv3d_deltaX_CrossAdd(streams, index, length, deltaY,OH,OW, W,FH,FW, deltaX,IH,IW, IC,OC, sh,sw,ph,pw,\
			GN, GMr, GK, oc_index, next_j_index);}}

#endif

//only used for IC <= (8 or 16)
void dconv3d_deltaX_CrossAdd(jlong *streams, int &index, int length,
	const float*  deltaY, int OH, int OW,
	const float*       W, int FH, int FW,
	      float*  deltaX, int IH, int IW,
	int IC, int OC,
	int sh, int sw, int ph, int pw,
	int GN, int GM, int GK,
	int oc_index, int j_index)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)streams[index];
	index = (index + 1) % length;

	if ((GN > 255) && (GM > 31) && !(GK & 7)) {//16*16 = 256, 2*16 = 32
		crossAdd_k16_2(stream, 4, oc_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, sh, sw, ph, pw, GN, GM);
		dconv3d_dX_CrossAdd_Branch(255, 31); return;
	}
	if ((GN > 127) && (GM > 31) && !(GK & 7)) {//8*16 = 128, 2*16 = 32
		crossAdd_k82(stream, 4, oc_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, sh, sw, ph, pw, GN, GM);
		dconv3d_dX_CrossAdd_Branch(127, 31); return;
	}

	if ((GN > 127) && (GM > 15)) {//16*8 = 128, 2*8 = 16
		crossAdd_k16_2(stream, 3, oc_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, sh, sw, ph, pw, GN, GM);
		dconv3d_dX_CrossAdd_Branch(127, 15); return;
	}
	if ((GN > 63) && (GM > 15)) {//8*8 = 64, 2*8 = 16
		crossAdd_k82(stream, 3, oc_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, sh, sw, ph, pw, GN, GM);
		dconv3d_dX_CrossAdd_Branch(63, 15); return;
	}

	if ((GN > 31) && (GM > 15)) {//4*8 = 32
		crossAdd_k42(stream, 3, oc_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, sh, sw, ph, pw, GN, GM);
		dconv3d_dX_CrossAdd_Branch(31, 15); return;
	}
	if ((GN > 15) && (GM > 15)) {//2*8 = 16
		crossAdd_k22(stream, 3, oc_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, sh, sw, ph, pw, GN, GM);
		dconv3d_dX_CrossAdd_Branch(15, 15); return;
	}
	if ((GN > 7) && (GM > 7) && !(GK &7)) {//1*8 = 8
		crossAdd_k11(stream, 3, oc_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, sh, sw, ph, pw, GN, GM);
		dconv3d_dX_CrossAdd_Branch(7, 7); return;
	}
	crossAdd_k11(stream, 2, oc_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, sh, sw, ph, pw, GN, GM);
}

#endif

#endif



//Im2col-Winograd: F(n, r)
#ifndef IM2COL_WINOGRAD_AREA
#define IM2COL_WINOGRAD_AREA

#ifndef IM2COL_WINOGRAD_MICRO
#define IM2COL_WINOGRAD_MICRO

#define im2col_winograd_ZeroPadding_s1_Gate(GNr, GMr) {\
	if(GNr && GMr){\
		int next_ic_index = (GN - GNr);\
		int next_j_index  = (GM - GMr);\
		dconv3d_deltaX_ZeroPadding_s1(streams,index,length, deltaY,OH,OW, W,FH,FW, deltaX,IH,IW, N,IC,OC, ph,pw,\
			GNr, GM, GK, next_ic_index, 0);\
		dconv3d_deltaX_ZeroPadding_s1(streams,index,length, deltaY,OH,OW, W,FH,FW, deltaX,IH,IW, N,IC,OC, ph,pw,\
			GN, GMr, GK, 0, next_j_index);\
		dconv3d_deltaX_ZeroPadding_s1(streams,index,length, deltaY,OH,OW, W,FH,FW, deltaX,IH,IW, N,IC,OC, ph,pw,\
			GNr, GMr, GK, next_ic_index, next_j_index);}\
	else if(GNr){\
		int next_ic_index = (GN - GNr);\
		dconv3d_deltaX_ZeroPadding_s1(streams,index,length, deltaY,OH,OW, W,FH,FW, deltaX,IH,IW, N,IC,OC, ph,pw,\
			GNr, GM, GK, next_ic_index, 0);}\
	else if(GMr){\
		int next_j_index = (GM - GMr);\
		dconv3d_deltaX_ZeroPadding_s1(streams,index,length, deltaY,OH,OW, W,FH,FW, deltaX,IH,IW, N,IC,OC, ph,pw,\
			GN, GMr, GK, 0, next_j_index);}}

#define im2col_winograd_ZeroPadding_s1_Gate_tex(GNr, GMr) {\
	if(GNr && GMr){\
		int next_ic_index = (GN - GNr);\
		int next_j_index  = (GM - GMr);\
		dconv3d_deltaX_ZeroPadding_s1_texture(streams,index,length, texDy,deltaY,OH,OW, W,FH,FW, deltaX,IH,IW, N,IC,OC, ph,pw,\
			GNr, GM, GK, next_ic_index, 0);\
		dconv3d_deltaX_ZeroPadding_s1_texture(streams,index,length, texDy,deltaY,OH,OW, W,FH,FW, deltaX,IH,IW, N,IC,OC, ph,pw,\
			GN, GMr, GK, 0, next_j_index);\
		dconv3d_deltaX_ZeroPadding_s1_texture(streams,index,length, texDy,deltaY,OH,OW, W,FH,FW, deltaX,IH,IW, N,IC,OC, ph,pw,\
			GNr, GMr, GK, next_ic_index, next_j_index);}\
	else if(GNr){\
		int next_ic_index = (GN - GNr);\
		dconv3d_deltaX_ZeroPadding_s1_texture(streams,index,length, texDy,deltaY,OH,OW, W,FH,FW, deltaX,IH,IW, N,IC,OC, ph,pw,\
			GNr, GM, GK, next_ic_index, 0);}\
	else if(GMr){\
		int next_j_index = (GM - GMr);\
		dconv3d_deltaX_ZeroPadding_s1_texture(streams,index,length, texDy,deltaY,OH,OW, W,FH,FW, deltaX,IH,IW, N,IC,OC, ph,pw,\
			GN, GMr, GK, 0, next_j_index);}}


#endif


#ifndef IM2COL_WINOGRAD_S8
#define IM2COL_WINOGRAD_S8

#define __dconv3D_deltaX_Im2col_Winograd_s8_tex(env, streams, index, length, deltaY, texDy, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, OC, ph, pw)\
	dconv3d_dX_im2col_Winograd_s8_texture(env,streams,index,length, deltaY,texDy,OH,OW, W,FH,FW, deltaX,IH,IW, N,IC,OC, ph,pw,\
		GET_GN_ZeroPadding(IC), GET_GM_ZeroPadding(N, IH, IW));

//(1) OC % 8 == 0
//(2) env != NULL: useTexture
//(3) GM = N * IH * IW;  GM >= 4, GM % 4 == 0
//(4) GN = IC;           GN >= 4, GN % 4 == 0
//(5) (FH, FW) belongs to [3, 4, 5, 6, 7]
//(6) GN: IC   >= 64
//(7) GM: N*IH >= 32
bool dconv3d_dX_im2col_Winograd_s8_texture(JNIEnv *env, jlong* streams, int &index, int length,
	const float* deltaY, cudaTextureObject_t texDy, int OH, int OW,
	const float*      W, int FH, int FW,
	      float* deltaX, int IH, int IW,
	int N, int IC, int OC,
	int ph, int pw,//sh = sw = 1
	int GN, int GM)
{
	int GNr, GMr;

	//==============[FW = 2x + 1]=======================================================================================================================================
	if ((FW == 3) && (IW >= 6)) {//F(6, 3)
		if (FH == 3) { deconv3D_dX_winograd_s8_W3_64x32R_tex<3>(streams, index, length, deltaY, texDy, OH, OW, W, deltaX, IH, IW, N, IC, OC, ph, pw, GN, GM, GMr, GNr); goto END_TRUE; }//3*3
		if (FH == 5) { deconv3D_dX_winograd_s8_W3_64x32R_tex<5>(streams, index, length, deltaY, texDy, OH, OW, W, deltaX, IH, IW, N, IC, OC, ph, pw, GN, GM, GMr, GNr); goto END_TRUE; }
		if (FH == 7) { deconv3D_dX_winograd_s8_W3_64x32R_tex<7>(streams, index, length, deltaY, texDy, OH, OW, W, deltaX, IH, IW, N, IC, OC, ph, pw, GN, GM, GMr, GNr); goto END_TRUE; }
		if (FH == 2) { deconv3D_dX_winograd_s8_W3_64x32R_tex<2>(streams, index, length, deltaY, texDy, OH, OW, W, deltaX, IH, IW, N, IC, OC, ph, pw, GN, GM, GMr, GNr); goto END_TRUE; }
		if (FH == 4) { deconv3D_dX_winograd_s8_W3_64x32R_tex<4>(streams, index, length, deltaY, texDy, OH, OW, W, deltaX, IH, IW, N, IC, OC, ph, pw, GN, GM, GMr, GNr); goto END_TRUE; }
		if (FH == 6) { deconv3D_dX_winograd_s8_W3_64x32R_tex<6>(streams, index, length, deltaY, texDy, OH, OW, W, deltaX, IH, IW, N, IC, OC, ph, pw, GN, GM, GMr, GNr); goto END_TRUE; }
		goto END_FALSE;
	}

	if ((FW == 5) && (IW >= 4)) {//F(4, 5)
		if (FH == 5) { deconv3D_dX_winograd_s8_W5_64x32R_tex<5>(streams, index, length, deltaY, texDy, OH, OW, W, deltaX, IH, IW, N, IC, OC, ph, pw, GN, GM, GMr, GNr); goto END_TRUE; }//5*5
		if (FH == 3) { deconv3D_dX_winograd_s8_W5_64x32R_tex<3>(streams, index, length, deltaY, texDy, OH, OW, W, deltaX, IH, IW, N, IC, OC, ph, pw, GN, GM, GMr, GNr); goto END_TRUE; }
		if (FH == 7) { deconv3D_dX_winograd_s8_W5_64x32R_tex<7>(streams, index, length, deltaY, texDy, OH, OW, W, deltaX, IH, IW, N, IC, OC, ph, pw, GN, GM, GMr, GNr); goto END_TRUE; }
		if (FH == 2) { deconv3D_dX_winograd_s8_W5_64x32R_tex<2>(streams, index, length, deltaY, texDy, OH, OW, W, deltaX, IH, IW, N, IC, OC, ph, pw, GN, GM, GMr, GNr); goto END_TRUE; }
		if (FH == 4) { deconv3D_dX_winograd_s8_W5_64x32R_tex<4>(streams, index, length, deltaY, texDy, OH, OW, W, deltaX, IH, IW, N, IC, OC, ph, pw, GN, GM, GMr, GNr); goto END_TRUE; }
		if (FH == 6) { deconv3D_dX_winograd_s8_W5_64x32R_tex<6>(streams, index, length, deltaY, texDy, OH, OW, W, deltaX, IH, IW, N, IC, OC, ph, pw, GN, GM, GMr, GNr); goto END_TRUE; }
		goto END_FALSE;
	}

	if ((FW == 7) && (IW >= 2)) {//F(2, 7)
		if (FH == 7) { deconv3D_dX_winograd_s8_W7_64x32R_tex<7>(streams, index, length, deltaY, texDy, OH, OW, W, deltaX, IH, IW, N, IC, OC, ph, pw, GN, GM, GMr, GNr); goto END_TRUE; }//7*7
		if (FH == 3) { deconv3D_dX_winograd_s8_W7_64x32R_tex<3>(streams, index, length, deltaY, texDy, OH, OW, W, deltaX, IH, IW, N, IC, OC, ph, pw, GN, GM, GMr, GNr); goto END_TRUE; }
		if (FH == 5) { deconv3D_dX_winograd_s8_W7_64x32R_tex<5>(streams, index, length, deltaY, texDy, OH, OW, W, deltaX, IH, IW, N, IC, OC, ph, pw, GN, GM, GMr, GNr); goto END_TRUE; }
		if (FH == 2) { deconv3D_dX_winograd_s8_W7_64x32R_tex<2>(streams, index, length, deltaY, texDy, OH, OW, W, deltaX, IH, IW, N, IC, OC, ph, pw, GN, GM, GMr, GNr); goto END_TRUE; }
		if (FH == 4) { deconv3D_dX_winograd_s8_W7_64x32R_tex<4>(streams, index, length, deltaY, texDy, OH, OW, W, deltaX, IH, IW, N, IC, OC, ph, pw, GN, GM, GMr, GNr); goto END_TRUE; }
		if (FH == 6) { deconv3D_dX_winograd_s8_W7_64x32R_tex<6>(streams, index, length, deltaY, texDy, OH, OW, W, deltaX, IH, IW, N, IC, OC, ph, pw, GN, GM, GMr, GNr); goto END_TRUE; }
		goto END_FALSE;
	}
	
	//==============[FW = 2x]===========================================================================================================================================
	if ((FW == 2) && (IW >= 7)) {//F(7, 2)
		if (FH == 2) { deconv3D_dX_winograd_s8_W2_64x32R_tex<2>(streams, index, length, deltaY, texDy, OH, OW, W, deltaX, IH, IW, N, IC, OC, ph, pw, GN, GM, GMr, GNr); goto END_TRUE; }//2*2
		if (FH == 4) { deconv3D_dX_winograd_s8_W2_64x32R_tex<4>(streams, index, length, deltaY, texDy, OH, OW, W, deltaX, IH, IW, N, IC, OC, ph, pw, GN, GM, GMr, GNr); goto END_TRUE; }
		if (FH == 6) { deconv3D_dX_winograd_s8_W2_64x32R_tex<6>(streams, index, length, deltaY, texDy, OH, OW, W, deltaX, IH, IW, N, IC, OC, ph, pw, GN, GM, GMr, GNr); goto END_TRUE; }
		if (FH == 3) { deconv3D_dX_winograd_s8_W2_64x32R_tex<3>(streams, index, length, deltaY, texDy, OH, OW, W, deltaX, IH, IW, N, IC, OC, ph, pw, GN, GM, GMr, GNr); goto END_TRUE; }
		if (FH == 5) { deconv3D_dX_winograd_s8_W2_64x32R_tex<5>(streams, index, length, deltaY, texDy, OH, OW, W, deltaX, IH, IW, N, IC, OC, ph, pw, GN, GM, GMr, GNr); goto END_TRUE; }
		if (FH == 7) { deconv3D_dX_winograd_s8_W2_64x32R_tex<7>(streams, index, length, deltaY, texDy, OH, OW, W, deltaX, IH, IW, N, IC, OC, ph, pw, GN, GM, GMr, GNr); goto END_TRUE; }
		goto END_FALSE;
	}

	if ((FW == 4) && (IW >= 5)) {//F(5, 4)
		if (FH == 4) { deconv3D_dX_winograd_s8_W4_64x32R_tex<4>(streams, index, length, deltaY, texDy, OH, OW, W, deltaX, IH, IW, N, IC, OC, ph, pw, GN, GM, GMr, GNr); goto END_TRUE; }//4*4
		if (FH == 2) { deconv3D_dX_winograd_s8_W4_64x32R_tex<2>(streams, index, length, deltaY, texDy, OH, OW, W, deltaX, IH, IW, N, IC, OC, ph, pw, GN, GM, GMr, GNr); goto END_TRUE; }
		if (FH == 6) { deconv3D_dX_winograd_s8_W4_64x32R_tex<6>(streams, index, length, deltaY, texDy, OH, OW, W, deltaX, IH, IW, N, IC, OC, ph, pw, GN, GM, GMr, GNr); goto END_TRUE; }
		if (FH == 3) { deconv3D_dX_winograd_s8_W4_64x32R_tex<3>(streams, index, length, deltaY, texDy, OH, OW, W, deltaX, IH, IW, N, IC, OC, ph, pw, GN, GM, GMr, GNr); goto END_TRUE; }
		if (FH == 5) { deconv3D_dX_winograd_s8_W4_64x32R_tex<5>(streams, index, length, deltaY, texDy, OH, OW, W, deltaX, IH, IW, N, IC, OC, ph, pw, GN, GM, GMr, GNr); goto END_TRUE; }
		if (FH == 7) { deconv3D_dX_winograd_s8_W4_64x32R_tex<7>(streams, index, length, deltaY, texDy, OH, OW, W, deltaX, IH, IW, N, IC, OC, ph, pw, GN, GM, GMr, GNr); goto END_TRUE; }
		goto END_FALSE;
	}

	if ((FW == 6) && (IW >= 3)) {//F(6, 3)
		if (FH == 6) { deconv3D_dX_winograd_s8_W6_64x32R_tex<6>(streams, index, length, deltaY, texDy, OH, OW, W, deltaX, IH, IW, N, IC, OC, ph, pw, GN, GM, GMr, GNr); goto END_TRUE; }//6*6
		if (FH == 2) { deconv3D_dX_winograd_s8_W6_64x32R_tex<2>(streams, index, length, deltaY, texDy, OH, OW, W, deltaX, IH, IW, N, IC, OC, ph, pw, GN, GM, GMr, GNr); goto END_TRUE; }
		if (FH == 4) { deconv3D_dX_winograd_s8_W6_64x32R_tex<4>(streams, index, length, deltaY, texDy, OH, OW, W, deltaX, IH, IW, N, IC, OC, ph, pw, GN, GM, GMr, GNr); goto END_TRUE; }
		if (FH == 3) { deconv3D_dX_winograd_s8_W6_64x32R_tex<3>(streams, index, length, deltaY, texDy, OH, OW, W, deltaX, IH, IW, N, IC, OC, ph, pw, GN, GM, GMr, GNr); goto END_TRUE; }
		if (FH == 5) { deconv3D_dX_winograd_s8_W6_64x32R_tex<5>(streams, index, length, deltaY, texDy, OH, OW, W, deltaX, IH, IW, N, IC, OC, ph, pw, GN, GM, GMr, GNr); goto END_TRUE; }
		if (FH == 7) { deconv3D_dX_winograd_s8_W6_64x32R_tex<7>(streams, index, length, deltaY, texDy, OH, OW, W, deltaX, IH, IW, N, IC, OC, ph, pw, GN, GM, GMr, GNr); goto END_TRUE; }
		goto END_FALSE;
	}

END_FALSE:
	return false;

END_TRUE:
	const int GK = FH * FW * OC;
	if (index > 0) index--;//save L2 cache: texDy != deltaY
	if (env) { im2col_winograd_ZeroPadding_s1_Gate_tex(GNr, GMr); }
	else { im2col_winograd_ZeroPadding_s1_Gate(GNr, GMr); }
	return true;
}


#endif


#ifndef IM2COL_WINOGRAD_S16
#define IM2COL_WINOGRAD_S16

#define __dconv3D_deltaX_Im2col_Winograd_s16_tex(env, streams, index, length, deltaY, texDy, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, OC, ph, pw)\
	dconv3d_dX_im2col_Winograd_s16_texture(env,streams,index,length, deltaY,texDy,OH,OW, W,FH,FW, deltaX,IH,IW, N,IC,OC, ph,pw,\
		GET_GN_ZeroPadding(IC), GET_GM_ZeroPadding(N, IH, IW));

//(1) OC % 8 == 0
//(2) env != NULL: useTexture
//(3) GM = N * IH * IW;  GM >= 4, GM % 4 == 0
//(4) GN = IC;           GN >= 4, GN % 4 == 0
//(5) (FH, FW) belongs to [3, 4, 5, 6, 7]
//(6) GN: IC   >= 64
//(7) GM: N*IH >= 32
bool dconv3d_dX_im2col_Winograd_s16_texture(JNIEnv *env, jlong* streams, int &index, int length,
	const float* deltaY, cudaTextureObject_t texDy, int OH, int OW,
	const float*      W, int FH, int FW,
	      float* deltaX, int IH, int IW,
	int N, int IC, int OC,
	int ph, int pw,//sh = sw = 1
	int GN, int GM)
{
	int GNr, GMr;

	//==============[FW = 2x + 1]=======================================================================================================================================
	if ((FW == 7) && (IW >= 10) && (pw >= 3) && (OW - IW - pw + 3 >= 0)) {
		if (FH == 7) { if (!(IC & 63)) deconv3D_dX_Winograd_s16_W7_64x32_p3_tex<7>(streams, index, length, deltaY, texDy, OH, OW, W, deltaX, IH, IW, N, IC, OC, ph, pw, GN, GM, GNr, GMr);
			           else            deconv3D_dX_Winograd_s16_W7_32x32_p3_tex<7>(streams, index, length, deltaY, texDy, OH, OW, W, deltaX, IH, IW, N, IC, OC, ph, pw, GN, GM, GNr, GMr); goto END_TRUE; }//7*7
		if (FH == 3) { if (!(IC & 63)) deconv3D_dX_Winograd_s16_W7_64x32_p3_tex<3>(streams, index, length, deltaY, texDy, OH, OW, W, deltaX, IH, IW, N, IC, OC, ph, pw, GN, GM, GNr, GMr);
		  		       else            deconv3D_dX_Winograd_s16_W7_32x32_p3_tex<3>(streams, index, length, deltaY, texDy, OH, OW, W, deltaX, IH, IW, N, IC, OC, ph, pw, GN, GM, GNr, GMr); goto END_TRUE; }
		if (FH == 5) { if (!(IC & 63)) deconv3D_dX_Winograd_s16_W7_64x32_p3_tex<5>(streams, index, length, deltaY, texDy, OH, OW, W, deltaX, IH, IW, N, IC, OC, ph, pw, GN, GM, GNr, GMr);
				       else            deconv3D_dX_Winograd_s16_W7_32x32_p3_tex<5>(streams, index, length, deltaY, texDy, OH, OW, W, deltaX, IH, IW, N, IC, OC, ph, pw, GN, GM, GNr, GMr); goto END_TRUE; }
		if (FH == 9) { if (!(IC & 63)) deconv3D_dX_Winograd_s16_W7_64x32_p3_tex<9>(streams, index, length, deltaY, texDy, OH, OW, W, deltaX, IH, IW, N, IC, OC, ph, pw, GN, GM, GNr, GMr);
				       else            deconv3D_dX_Winograd_s16_W7_32x32_p3_tex<9>(streams, index, length, deltaY, texDy, OH, OW, W, deltaX, IH, IW, N, IC, OC, ph, pw, GN, GM, GNr, GMr); goto END_TRUE; }
		if (FH == 2) { if (!(IC & 63)) deconv3D_dX_Winograd_s16_W7_64x32_p3_tex<2>(streams, index, length, deltaY, texDy, OH, OW, W, deltaX, IH, IW, N, IC, OC, ph, pw, GN, GM, GNr, GMr);
				       else            deconv3D_dX_Winograd_s16_W7_32x32_p3_tex<2>(streams, index, length, deltaY, texDy, OH, OW, W, deltaX, IH, IW, N, IC, OC, ph, pw, GN, GM, GNr, GMr); goto END_TRUE; }
		if (FH == 4) { if (!(IC & 63)) deconv3D_dX_Winograd_s16_W7_64x32_p3_tex<4>(streams, index, length, deltaY, texDy, OH, OW, W, deltaX, IH, IW, N, IC, OC, ph, pw, GN, GM, GNr, GMr);
				       else            deconv3D_dX_Winograd_s16_W7_32x32_p3_tex<4>(streams, index, length, deltaY, texDy, OH, OW, W, deltaX, IH, IW, N, IC, OC, ph, pw, GN, GM, GNr, GMr); goto END_TRUE; }
		if (FH == 6) { if (!(IC & 63)) deconv3D_dX_Winograd_s16_W7_64x32_p3_tex<6>(streams, index, length, deltaY, texDy, OH, OW, W, deltaX, IH, IW, N, IC, OC, ph, pw, GN, GM, GNr, GMr);
				       else            deconv3D_dX_Winograd_s16_W7_32x32_p3_tex<6>(streams, index, length, deltaY, texDy, OH, OW, W, deltaX, IH, IW, N, IC, OC, ph, pw, GN, GM, GNr, GMr); goto END_TRUE; }
		if (FH == 8) { if (!(IC & 63)) deconv3D_dX_Winograd_s16_W7_64x32_p3_tex<8>(streams, index, length, deltaY, texDy, OH, OW, W, deltaX, IH, IW, N, IC, OC, ph, pw, GN, GM, GNr, GMr);
				       else            deconv3D_dX_Winograd_s16_W7_32x32_p3_tex<8>(streams, index, length, deltaY, texDy, OH, OW, W, deltaX, IH, IW, N, IC, OC, ph, pw, GN, GM, GNr, GMr); goto END_TRUE; }
		goto END_FALSE;
	}

	if ((FW == 9) && (IW >= 8) && (pw >= 4) && (OW - IW - pw + 4 >= 0)) {
		if (FH == 9) { if (!(IC & 63)) deconv3D_dX_Winograd_s16_W9_64x32_p4_tex<9>(streams, index, length, deltaY, texDy, OH, OW, W, deltaX, IH, IW, N, IC, OC, ph, pw, GN, GM, GNr, GMr);
			           else            deconv3D_dX_Winograd_s16_W9_32x32_p4_tex<9>(streams, index, length, deltaY, texDy, OH, OW, W, deltaX, IH, IW, N, IC, OC, ph, pw, GN, GM, GNr, GMr); goto END_TRUE; }//9*9
		if (FH == 5) { if (!(IC & 63)) deconv3D_dX_Winograd_s16_W9_64x32_p4_tex<5>(streams, index, length, deltaY, texDy, OH, OW, W, deltaX, IH, IW, N, IC, OC, ph, pw, GN, GM, GNr, GMr);
			           else            deconv3D_dX_Winograd_s16_W9_32x32_p4_tex<5>(streams, index, length, deltaY, texDy, OH, OW, W, deltaX, IH, IW, N, IC, OC, ph, pw, GN, GM, GNr, GMr); goto END_TRUE; }
		if (FH == 7) { if (!(IC & 63)) deconv3D_dX_Winograd_s16_W9_64x32_p4_tex<7>(streams, index, length, deltaY, texDy, OH, OW, W, deltaX, IH, IW, N, IC, OC, ph, pw, GN, GM, GNr, GMr);
			           else            deconv3D_dX_Winograd_s16_W9_32x32_p4_tex<7>(streams, index, length, deltaY, texDy, OH, OW, W, deltaX, IH, IW, N, IC, OC, ph, pw, GN, GM, GNr, GMr); goto END_TRUE; }
		if (FH == 3) { if (!(IC & 63)) deconv3D_dX_Winograd_s16_W9_64x32_p4_tex<3>(streams, index, length, deltaY, texDy, OH, OW, W, deltaX, IH, IW, N, IC, OC, ph, pw, GN, GM, GNr, GMr);
		               else            deconv3D_dX_Winograd_s16_W9_32x32_p4_tex<3>(streams, index, length, deltaY, texDy, OH, OW, W, deltaX, IH, IW, N, IC, OC, ph, pw, GN, GM, GNr, GMr); goto END_TRUE; }
		if (FH == 2) { if (!(IC & 63)) deconv3D_dX_Winograd_s16_W9_64x32_p4_tex<2>(streams, index, length, deltaY, texDy, OH, OW, W, deltaX, IH, IW, N, IC, OC, ph, pw, GN, GM, GNr, GMr);
			           else            deconv3D_dX_Winograd_s16_W9_32x32_p4_tex<2>(streams, index, length, deltaY, texDy, OH, OW, W, deltaX, IH, IW, N, IC, OC, ph, pw, GN, GM, GNr, GMr); goto END_TRUE; }
		if (FH == 4) { if (!(IC & 63)) deconv3D_dX_Winograd_s16_W9_64x32_p4_tex<4>(streams, index, length, deltaY, texDy, OH, OW, W, deltaX, IH, IW, N, IC, OC, ph, pw, GN, GM, GNr, GMr);
			           else            deconv3D_dX_Winograd_s16_W9_32x32_p4_tex<4>(streams, index, length, deltaY, texDy, OH, OW, W, deltaX, IH, IW, N, IC, OC, ph, pw, GN, GM, GNr, GMr); goto END_TRUE; }
		if (FH == 6) { if (!(IC & 63)) deconv3D_dX_Winograd_s16_W9_64x32_p4_tex<6>(streams, index, length, deltaY, texDy, OH, OW, W, deltaX, IH, IW, N, IC, OC, ph, pw, GN, GM, GNr, GMr);
			           else            deconv3D_dX_Winograd_s16_W9_32x32_p4_tex<6>(streams, index, length, deltaY, texDy, OH, OW, W, deltaX, IH, IW, N, IC, OC, ph, pw, GN, GM, GNr, GMr); goto END_TRUE; }
		if (FH == 8) { if (!(IC & 63)) deconv3D_dX_Winograd_s16_W9_64x32_p4_tex<8>(streams, index, length, deltaY, texDy, OH, OW, W, deltaX, IH, IW, N, IC, OC, ph, pw, GN, GM, GNr, GMr);
			           else            deconv3D_dX_Winograd_s16_W9_32x32_p4_tex<8>(streams, index, length, deltaY, texDy, OH, OW, W, deltaX, IH, IW, N, IC, OC, ph, pw, GN, GM, GNr, GMr); goto END_TRUE; }
		goto END_FALSE;
	}

	//==============[FW = 2x]=========================================================================================================================================
	if ((FW == 8) && (IW >= 9) && (pw >= 3) && (OW - IW - pw + 5 >= 0)) {
		if (FH == 8) { if (!(IC & 63)) deconv3D_dX_Winograd_s16_W8_64x32_p3_tex<8>(streams, index, length, deltaY, texDy, OH, OW, W, deltaX, IH, IW, N, IC, OC, ph, pw, GN, GM, GNr, GMr);
			           else            deconv3D_dX_Winograd_s16_W8_32x32_p3_tex<8>(streams, index, length, deltaY, texDy, OH, OW, W, deltaX, IH, IW, N, IC, OC, ph, pw, GN, GM, GNr, GMr); goto END_TRUE; }//8*8
		if (FH == 2) { if (!(IC & 63)) deconv3D_dX_Winograd_s16_W8_64x32_p3_tex<2>(streams, index, length, deltaY, texDy, OH, OW, W, deltaX, IH, IW, N, IC, OC, ph, pw, GN, GM, GNr, GMr);
			           else            deconv3D_dX_Winograd_s16_W8_32x32_p3_tex<2>(streams, index, length, deltaY, texDy, OH, OW, W, deltaX, IH, IW, N, IC, OC, ph, pw, GN, GM, GNr, GMr); goto END_TRUE; }
		if (FH == 4) { if (!(IC & 63)) deconv3D_dX_Winograd_s16_W8_64x32_p3_tex<4>(streams, index, length, deltaY, texDy, OH, OW, W, deltaX, IH, IW, N, IC, OC, ph, pw, GN, GM, GNr, GMr);
			           else            deconv3D_dX_Winograd_s16_W8_32x32_p3_tex<4>(streams, index, length, deltaY, texDy, OH, OW, W, deltaX, IH, IW, N, IC, OC, ph, pw, GN, GM, GNr, GMr); goto END_TRUE; }
		if (FH == 6) { if (!(IC & 63)) deconv3D_dX_Winograd_s16_W8_64x32_p3_tex<6>(streams, index, length, deltaY, texDy, OH, OW, W, deltaX, IH, IW, N, IC, OC, ph, pw, GN, GM, GNr, GMr);
			           else            deconv3D_dX_Winograd_s16_W8_32x32_p3_tex<6>(streams, index, length, deltaY, texDy, OH, OW, W, deltaX, IH, IW, N, IC, OC, ph, pw, GN, GM, GNr, GMr); goto END_TRUE; }
		if (FH == 9) { if (!(IC & 63)) deconv3D_dX_Winograd_s16_W8_64x32_p3_tex<9>(streams, index, length, deltaY, texDy, OH, OW, W, deltaX, IH, IW, N, IC, OC, ph, pw, GN, GM, GNr, GMr);
			           else            deconv3D_dX_Winograd_s16_W8_32x32_p3_tex<9>(streams, index, length, deltaY, texDy, OH, OW, W, deltaX, IH, IW, N, IC, OC, ph, pw, GN, GM, GNr, GMr); goto END_TRUE; }
		if (FH == 5) { if (!(IC & 63)) deconv3D_dX_Winograd_s16_W8_64x32_p3_tex<5>(streams, index, length, deltaY, texDy, OH, OW, W, deltaX, IH, IW, N, IC, OC, ph, pw, GN, GM, GNr, GMr);
			           else            deconv3D_dX_Winograd_s16_W8_32x32_p3_tex<5>(streams, index, length, deltaY, texDy, OH, OW, W, deltaX, IH, IW, N, IC, OC, ph, pw, GN, GM, GNr, GMr); goto END_TRUE; }
		if (FH == 7) { if (!(IC & 63)) deconv3D_dX_Winograd_s16_W8_64x32_p3_tex<7>(streams, index, length, deltaY, texDy, OH, OW, W, deltaX, IH, IW, N, IC, OC, ph, pw, GN, GM, GNr, GMr);
			           else            deconv3D_dX_Winograd_s16_W8_32x32_p3_tex<7>(streams, index, length, deltaY, texDy, OH, OW, W, deltaX, IH, IW, N, IC, OC, ph, pw, GN, GM, GNr, GMr); goto END_TRUE; }
		if (FH == 3) { if (!(IC & 63)) deconv3D_dX_Winograd_s16_W8_64x32_p3_tex<3>(streams, index, length, deltaY, texDy, OH, OW, W, deltaX, IH, IW, N, IC, OC, ph, pw, GN, GM, GNr, GMr);
			           else            deconv3D_dX_Winograd_s16_W8_32x32_p3_tex<3>(streams, index, length, deltaY, texDy, OH, OW, W, deltaX, IH, IW, N, IC, OC, ph, pw, GN, GM, GNr, GMr); goto END_TRUE; }
		goto END_FALSE;
	}

END_FALSE:
	return false;

END_TRUE:
	const int GK = FH * FW * OC;
	if (index > 0) index--;//save L2 cache: texDy != deltaY
	if (env) { im2col_winograd_ZeroPadding_s1_Gate_tex(GNr, GMr); }
	else { im2col_winograd_ZeroPadding_s1_Gate(GNr, GMr); }
	return true;
}

#endif

#endif

#endif//complie-area>>>>------------------------------------------------------------


#endif

