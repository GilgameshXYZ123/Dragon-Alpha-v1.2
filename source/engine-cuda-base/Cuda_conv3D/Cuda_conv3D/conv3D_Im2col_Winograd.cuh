#pragma once

#ifndef CONV_3D_IM2COL_WINOGRAD_H
#define CONV_3D_IM2COL_WINOGRAD_H

//process remainder of OW
#include "conv3D_GemmR_uernel_C.cuh"

//4 states
#include "conv3D_ori_Winograd_f2x3R.cuh"//FW = 3
#include "conv3D_Im2col_Winograd_s4_f3x2R.cuh"//FW = 2
#include "conv3D_Im2col_Winograd_s4_f2x3R.cuh"//FW = 3

//8 states
#include "conv3D_Im2col_Winograd_s8_f7x2R.cuh"//FW = 2
#include "conv3D_Im2col_Winograd_s8_f6x3R.cuh"//FW = 3
#include "conv3D_Im2col_Winograd_s8_f5x4R.cuh"//FW = 4
#include "conv3D_Im2col_Winograd_s8_f4x5R.cuh"//FW = 5
#include "conv3D_Im2col_Winograd_s8_f3x6R.cuh"//FW = 6
#include "conv3D_Im2col_Winograd_s8_f2x7R.cuh"//FW = 7

//16 states
#include "conv3D_Im2col_Winograd_sg_fAx7R.cuh"//FW = 7
#include "conv3D_Im2col_Winograd_sg_f9x8R.cuh"//FW = 8
#include "conv3D_Im2col_Winograd_sg_f8x9R.cuh"//FW = 9

#include "conv3D_WinogradV2_f6x3R.cuh"//FW = 3


//------[state = 8]------------------------------
#ifndef CONV_3D_WINOGRAD_S8_AREA
#define CONV_3D_WINOGRAD_S8_AREA

//FW = 2
#ifndef CONV_3D_WINOGRAD_S8_W2_64X32R_TEXTURE
#define CONV_3D_WINOGRAD_S8_W2_64X32R_TEXTURE

#ifndef CONV_3D_WINOGRAD_S8_W2_64X32R_TEXTURE_BENCH_MARK_RTX3060TI
#define CONV_3D_WINOGRAD_S8_W2_64X32R_TEXTURE_BENCH_MARK_RTX3060TI

//[0] for: Feature = (128, 128), [N, IC, OC] = [32, 128, 128]
//LB = 4: Size = 32, Time = 5.7782  msec, Performace = 11892.9 GFlop/s
//WB = 4: Size = 32, Time = 5.50876 msec, Performace = 12474.6 GFlop/s
//cuDNN-NCHW-implicit-prec: Size = 32, Time = 5.567 msec, Performance = 12344.1 GFlop/s
//cuDNN-NHWC-implicit-prec: Size = 32, Time = 5.699 msec, Performance = 12058.2 GFlop/s
//cuDNN-NHWC-GEMM-implicit: Size = 32, Time = 11.575 msec, Performance = 5936.9 GFlop/s

//[1] for: Feature = (112, 112), [N, IC, OC] = [128, 64, 64]
//LB = 4: Size = 24.5, Time = 4.75811 msec, Performace = 11057.6 GFlop/s
//WB = 4: Size = 24.5, Time = 4.52505 msec, Performace = 11627.1 GFlop/s
//cuDNN-NCHW-implicit-prec: Size = 24.5, Time = 5.909 msec, Performance = 8903.9 GFlop/s
//cuDNN-NHWC-implicit-prec: Size = 24.5, Time = 5.669 msec, Performance = 9280.9 GFlop/s
//cuDNN-NHWC-GEMM-implicit: Size = 24.5, Time = 16.97 msec, Performance = 3100.4 GFlop/s

//[2] for: Feature = (64, 64), [N, IC, OC] = [128, 128, 128]
//LB = 4: Size = 32, Time = 5.73614 msec, Performace = 11980.1 GFlop/s
//WB = 4: Size = 32, Time = 5.46193 msec, Performace = 12581.5 GFlop/s
//cuDNN-NCHW-implicit-prec: Size = 32, Time = 6.594 msec, Performance = 10421.5 GFlop/s
//cuDNN-NHWC-implicit-prec: Size = 32, Time = 5.657 msec, Performance = 12147.7 GFlop/s
//cuDNN-NHWC-GEMM-implicit: Size = 32, Time = 11.45 msec, Performance = 6001.7 GFlop/s

//[3] for: Feature = (56, 56), [N, IC, OC] = [128, 128, 128]
//LB = 4: Size = 24.5, Time = 4.31674 msec, Performace = 12188.2 GFlop/s
//WB = 4: Size = 24.5, Time = 4.0953  msec, Performace = 12847.2 GFlop/s
//cuDNN-NCHW-implicit-prec: Size = 24.5, Time = 5.088 msec, Performance = 10340.7 GFlop/s
//cuDNN-NHWC-implicit-prec: Size = 24.5, Time = 4.315 msec, Performance = 12193.1 GFlop/s
//cuDNN-NHWC-GEMM-implicit: Size = 24.5, Time = 8.875 msec, Performance = 5928.3 GFlop/s

//[4] for: Feature = (32, 32), [N, IC, OC] = [128, 256, 256]
//LB = 4: Size = 32, Time = 5.27239 msec, Performace = 13033.8 GFlop/s
//WB = 4: Size = 32, Time = 5.02119 msec, Performace = 13685.9 GFlop/s
//cuDNN-NCHW-implicit-prec: Size = 32, Time = 4.774 msec, Performance = 14394.5 GFlop/s
//cuDNN-NHWC-implicit-prec: Size = 32, Time = 4.859 msec, Performance = 14142.7 GFlop/s
//cuDNN-NHWC-GEMM-implicit: Size = 32, Time = 11.31 msec, Performance = 6076.0 GFlop/s

//[5] for: Feature = (28, 28), [N, IC, OC] = [128, 256, 256]
//LB = 4: Size = 24.5, Time = 3.86777 msec, Performace = 13603   GFlop/s
//WB = 4: Size = 24.5, Time = 3.62671 msec, Performace = 14507.2 GFlop/s
//cuDNN-NCHW-implicit-prec: Size = 24.5, Time = 3.686 msec, Performance = 14273.8 GFlop/s
//cuDNN-NHWC-implicit-prec: Size = 24.5, Time = 3.672 msec, Performance = 14328.3 GFlop/s
//cuDNN-NHWC-GEMM-implicit: Size = 24.5, Time = 8.695 msec, Performance = 6051.0 GFlop/s

//[6] for: Feature = (16, 16), [N, IC, OC] = [128, 512, 512]
//LB = 4: Size = 32, Time = 5.11727 msec, Performace = 13428.9 GFlop/s
//WB = 4: Size = 32, Time = 4.84119 msec, Performace = 14194.7 GFlop/s
//cuDNN-NCHW-implicit-prec: Size = 32, Time = 4.342 msec, Performance = 15826.7 GFlop/s
//cuDNN-NHWC-implicit-prec: Size = 32, Time = 4.303 msec, Performance = 15970.1 GFlop/s
//cuDNN-NHWC-GEMM-implicit: Size = 32, Time = 11.02 msec, Performance = 6235.9 GFlop/s

//[7] for: Feature = (14, 14), [N, IC, OC] = [128, 512, 512]
//LB = 4: Size = 24.5, Time = 3.72147 msec, Performace = 14137.8 GFlop/s
//WB = 4: Size = 24.5, Time = 3.44997 msec, Performace = 15250.4 GFlop/s
//cuDNN-NCHW-implicit-prec: Size = 24.5, Time = 3.642 msec, Performance = 14446.3 GFlop/s
//cuDNN-NHWC-implicit-prec: Size = 24.5, Time = 3.315 msec, Performance = 15871.3 GFlop/s
//cuDNN-NHWC-GEMM-implicit: Size = 24.5, Time = 8.46 msec, Performance = 6219.1 GFlop/s

//[8] for: Feature = (8, 8), [N, IC, OC] = [128, 1024, 1024]
//LB = 4: Size = 32, Time = 5.02566 msec, Performace = 13673.7 GFlop/s
//WB = 4: Size = 32, Time = 4.65363 msec, Performace = 14766.9 GFlop/s
//cuDNN-NCHW-implicit-prec: Size = 32, Time = 4.479 msec, Performance = 15342.6 GFlop/s
//cuDNN-NHWC-implicit-prec: Size = 32, Time = 4.817 msec, Performance = 14266.0 GFlop/s
//cuDNN-NHWC-GEMM-implicit: Size = 32, Time = 10.77 msec, Performance = 6380.6 GFlop/s

//[9] for: Feature = (7, 7), [N, IC, OC] = [128, 1024, 1024]
//LB = 4: Size = 24.5, Time = 3.64214 msec, Performace = 14445.7 GFlop/s
//WB = 4: Size = 24.5, Time = 3.31561 msec, Performace = 15868.4 GFlop/s
//cuDNN-NCHW-implicit-prec: Size = 24.5, Time = 3.497 msec, Performance = 15045.3 GFlop/s
//cuDNN-NHWC-implicit-prec: Size = 24.5, Time = 3.764 msec, Performance = 13978.0 GFlop/s
//cuDNN-NHWC-GEMM-implicit: Size = 24.5, Time = 8.23 msec, Performance = 6392.9 GFlop/s

#endif

//(1) FW = 2
//(2) sh = sw = 1
//(3) GN: (OC    ) % 64
//(4) GM: (N * OH) % 32
//(5) IC % 8 == 0
//(6) OW >= 7
#ifndef COMPILE
#define CONV_3D_WINOGRAD_S8_W2_64X32R_TEX
#endif
#ifndef CONV_3D_WINOGRAD_S8_W2_64X32R_TEX
#define CONV_3D_WINOGRAD_S8_W2_64X32R_TEX

template<int FH>
inline void conv3D_Winograd_s8_W2_64x32R_tex(jlong* streams, int &index, int length,
	const float*  X, cudaTextureObject_t texX, int IH, int IW,
	const float* CW,//[FH, FW, IC, OC]
	      float*  Y, int OH, int OW,
	int N, int IC, int OC,
	int ph, int pw,
	int GN, int GM, int &GNr, int &GMr)
{
	//======[Stage1 (OW % 7): Winograd_F(7, 2)]=========================================================================
	next_cudaStream(stream, streams, index, length);
	if ((pw <= 1) && (OW - IW - pw <= 0)) conv3dWinograd_f7x2_k64x224R_p1_tex(stream, texX, IH, IW, CW, FH, Y, OH, OW, N, IC, OC, ph, pw);
	else conv3dWinograd_f7x2_k64x224R_tex(stream, texX, IH, IW, CW, FH, Y, OH, OW, N, IC, OC, ph, pw);
	
	//======[Stage2 (OW % 6): Winograd_F(3+3, 2)]=======================================================================
	int OWr = OW % 7;
	if ((OWr >= 6) && !((N*OH) & 31)) {//Remainder: 6
		next_cudaStream(stream1, streams, index, length);
		conv3dWinograd_f3x2_k64x192R6C_tex(stream1, OW - OWr, texX, IH, IW, CW, FH, Y, OH, OW, N, IC, OC, ph, pw, OWr);
		OWr = OWr % 6;//OW % 6
	}

	//======[Stage3 (OW % 3): Winograd_F(3, 2)]=========================================================================
	if ((OWr >= 3) && !((N*OH) & 63)) {//Remainder: 3, 4, 5
		next_cudaStream(stream2, streams, index, length);
		conv3dWinograd_f3x2_k64x192RC_tex(stream2, OW - OWr, texX, IH, IW, CW, FH, Y, OH, OW, N, IC, OC, ph, pw, OWr);
		OWr = OWr % 3;//OW % 3
	}

	//======[Stage4: GEMM]==============================================================================================
	if (OWr > 0) {//Remainder: 1, 2
		const int GN64 = ((OC  ) >> 6 << 6);      //GN   = 64x
		const int GM32 = ((N*OH) >> 5 << 5) * OWr;//N*OH = 32x
		index = 0;//save L2 cache: texX != X
		conv3D_Gemm_32x32RC(streams, index, length,
			X, IH, IW, CW, FH, 2, Y, OH, OW, OWr, IC, OC, 1, 1, ph, pw,
			GN64, GM32, 0, 0, OW - OWr);
	}
	
	//=====[Process GNr and GMr]========================================================================================
	GNr = GN & 63;//64 channels
	GMr = ((N*OH) & 31)*OW;//32 groups
}

#endif

#endif


//FW = 3
#ifndef CONV_3D_WINOGRAD_S8_W3_64X32R_TEXTURE
#define CONV_3D_WINOGRAD_S8_W3_64X32R_TEXTURE

#ifndef CONV_3D_WINOGRAD_S8_W3_64X32R_TEXTURE_BENCK_MARK_RTX3060TI
#define CONV_3D_WINOGRAD_S8_W3_64X32R_TEXTURE_BENCK_MARK_RTX3060TI

//[0] for: Feature = (128, 128), [N, IC, OC] = [64, 64, 64]
//LB = 4: Size = 36, Time = 4.85967 msec, Performace = 15908.4 GFlop/s
//WB = 4: Size = 36, Time = 4.59537 msec, Performace = 16823.3 GFlop/s
//cuDNN-Winograd-Fused: Size = 36, Time = 5.482 msec, Performance = 14102.4 GFlop/s
//cuDNN-NCHW-GEMM-implicit-prec: Size = 36, Time = 7.361 msec, Performance = 10502.6 GFlop/s
//cuDNN-NHWC-GEMM-implicit-prec: Size = 36, Time = 6.159 msec, Performance = 12552.3 GFlop/s
//cuDNN-NHWC-GEMM-implicit: Size = 36, Time = 24.155 msec, Performance = 3200.6 GFlop/s

//[1] for: Feature = (96, 96), [N, IC, OC] = [128, 64, 64]
//LB = 4: Size = 40.5, Time = 5.51523 msec, Performace = 15769.6 GFlop/s
//WB = 4: Size = 40.5, Time = 5.07856 msec, Performace = 17125.5 GFlop/s
//cuDNN-Winograd-Fused: Size = 40.5, Time = 6.141 msec, Performance = 14162.7 GFlop/s
//cuDNN-NCHW-GEMM-implicit-prec: Size = 40.5, Time = 8.293 msec, Performance = 10487.5 GFlop/s
//cuDNN-NHWC-GEMM-implicit-prec: Size = 40.5, Time = 6.895 msec, Performance = 12613.9 GFlop/s
//cuDNN-NHWC-GEMM-implicit: Size = 40.5, Time = 27.015 msec, Performance = 3219.4 GFlop/s

//[2] for: Feature = (64, 64), [N, IC, OC] = [256, 64, 64]
//LB = 4: Size = 36, Time = 4.97699 msec, Performace = 15533.4 GFlop/s
//WB = 4: Size = 36, Time = 4.74423 msec, Performace = 16295.5 GFlop/s
//cuDNN-Winograd-Fused: Size = 36, Time = 5.396 msec, Performance = 14327.2 GFlop/s
//cuDNN-NCHW-GEMM-implicit-prec: Size = 36, Time = 5.22  msec, Performance = 14810.3 GFlop/s
//cuDNN-NHWC-GEMM-implicit-prec: Size = 36, Time = 5.238 msec, Performance = 14759.3 GFlop/s
//cuDNN-NHWC-GEMM-implicit: Size = 36, Time = 24.325 msec, Performance = 3178.2 GFlop/s

//[3] for: Feature = (48, 48), [N, IC, OC] = [128, 128, 128]
//LB = 4: Size = 40.5, Time = 5.04724 msec, Performace = 17231.8 GFlop/s
//WB = 4: Size = 40.5, Time = 4.69785 msec, Performace = 18513.4 GFlop/s
//cuDNN-Winograd-Fused: Size = 40.5, Time = 5.367 msec, Performance = 16205.2 GFlop/s
//cuDNN-NCHW-GEMM-implicit-prec: Size = 40.5, Time = 5.988 msec, Performance = 14524.6 GFlop/s
//cuDNN-NHWC-GEMM-implicit-prec: Size = 40.5, Time = 5.882 msec, Performance = 14786.3 GFlop/s
//cuDNN-NHWC-GEMM-implicit: Size = 40.5, Time = 14.36 msec, Performance = 6056.6 GFlop/s

//[4] for: Feature = (32, 32), [N, IC, OC] = [256, 128, 128]
//LB = 4: Size = 36, Time = 4.63619 msec, Performace = 16675.2 GFlop/s
//WB = 4: Size = 36, Time = 4.34027 msec, Performace = 17812.1 GFlop/s
//cuDNN-Winograd-Fused: Size = 36, Time = 5.469 msec, Performance = 14135.9 GFlop/s
//cuDNN-NCHW-GEMM-implicit-prec: Size = 36, Time = 5.268 msec, Performance = 14675.3 GFlop/s
//cuDNN-NHWC-GEMM-implicit-prec: Size = 36, Time = 5.243 msec, Performance = 14745.3 GFlop/s
//cuDNN-NHWC-GEMM-implicit: Size = 36, Time = 12.54 msec, Performance = 6165.0 GFlop/s

//[5] for: Feature = (24, 24), [N, IC, OC] = [128, 256, 256]
//LB = 4: Size = 40.5, Time = 4.74113 msec, Performace = 18344.4 GFlop/s
//WB = 4: Size = 40.5, Time = 4.46106 msec, Performace = 19496.1 GFlop/s
//cuDNN-Winograd-Fused: Size = 40.5, Time = 5.52985 msec, Performace = 15727.9 GFlop/s
//cuDNN-NCHW-GEMM-implicit-prec: Size = 40.5, Time = 5.797 msec, Performance = 15003.1 GFlop/s
//cuDNN-NHWC-GEMM-implicit-prec: Size = 40.5, Time = 5.45  msec, Performance = 15958.4 GFlop/s
//cuDNN-NHWC-GEMM-implicit: ize = 40.5, Time = 14.17 msec, Performance = 6137.8 GFlop/s

//[6] for: Feature = (16, 16), [N, IC, OC] = [256, 256, 256]
//LB = 4: Size = 36, Time = 4.60253 msec, Performace = 16797.2 GFlop/s
//WB = 4: Size = 36, Time = 4.29605 msec, Performace = 17995.5 GFlop/s
//cuDNN-Winograd-Fused: Size = 36, Time = 4.466 msec, Performance = 17310.7 GFlop/s
//cuDNN-NCHW-GEMM-implicit-prec: Size = 36, Time = 5.358 msec, Performance = 14428.8 GFlop/s
//cuDNN-NHWC-GEMM-implicit-prec: Size = 36, Time = 4.844 msec, Performance = 15959.8 GFlop/s
//cuDNN-NHWC-GEMM-implicit: Size = 36, Time = 12.785 msec, Performance = 6046.9 GFlop/s

//[7] for: Feature = (12, 12), [N, IC, OC] = [128, 512, 512]
//LB = 4: Size = 40.5, Time = 4.95093 msec, Performace = 17567 GFlop/s
//WB = 4: Size = 40.5, Time = 4.45236 msec, Performace = 19534.1 GFlop/s
//V2LB = 4: Size = 40.5, Time = 4.88319 msec, Performace = 17810.7 GFlop/s
//V2WB = 4: Size = 40.5, Time = 4.56674 msec, Performace = 19044.9 GFlop/ss
//cuDNN-Winograd-Fused: Size = 40.5, Time = 8.433 msec, Performance = 10313.4 GFlop/s
//cuDNN-NCHW-GEMM-implicit-prec: Size = 40.5, Time = 5.598 msec, Performance = 15536.5 GFlop/s
//cuDNN-NHWC-GEMM-implicit-prec: Size = 40.5, Time = 5.283 msec, Performance = 16462.8 GFlop/s
//cuDNN-NHWC-GEMM-implicit: Size = 40.5, Time = 14.18 msec, Performance = 6133.5 GFlop/s

//[8] for: Feature = (8, 8), [N, IC, OC] = [256, 512, 512]
//LB = 4: Size = 36, Time = 4.75015 msec, Performace = 16275.1 GFlop/s
//WB = 4: Size = 36, Time = 4.37165 msec, Performace = 17684.3 GFlop/s
//cuDNN-Winograd-Fused: Size = 36, Time = 4.722 msec, Performance = 16372.2 GFlop/s
//cuDNN-NCHW-GEMM-implicit-prec: Size = 36, Time = 4.752 msec, Performance = 16268.8 GFlop/s
//cuDNN-NHWC-GEMM-implicit-prec: Size = 36, Time = 4.562 msec, Performance = 16946.4 GFlop/s
//cuDNN-NHWC-GEMM-implicit: Size = 36, Time = 12.4 msec, Performance = 6234.6 GFlop/s

//[9] for: Feature = (6, 6), [N, IC, OC] = [128, 1024, 1024]
//LB = 4: Size = 40.5, Time = 4.99974 msec, Performace = 17395.5 GFlop/s
//WB = 4: Size = 40.5, Time = 4.53758 msec, Performace = 19167.3 GFlop/s
//V2LB = 4: Size = 40.5, Time = 4.66051 msec, Performace = 18661.7 GFlop/s
//V2WB = 4: Size = 40.5, Time = 4.16379 msec, Performace = 20888 GFlop/s
//cuDNN-Winograd-Fused: Size = 40.5, Time = 17.993 msec, Performance = 4833.7 GFlop/s
//cuDNN-NCHW-GEMM-implicit-prec: Size = 40.5, Time = 5.422 msec, Performance = 16040.8 GFlop/s
//cuDNN-NHWC-GEMM-implicit-prec: Size = 40.5, Time = 5.536 msec, Performance = 15710.5 GFlop/s
//cuDNN-NHWC-GEMM-implicit: Size = 40.5, Time = 14.185 msec, Performance = 6131.3 GFlop/s

#endif

//(1) FW = 3
//(2) sh = sw = 1
//(3) GN: (OC    ) % 64
//(4) GM: (N * OH) % 32
//(5) IC % 8 == 0
//(6) OW >= 6
#ifndef COMPILE
#define CONV_3D_WINOGRAD_S8_W3_64X32R_TEX
#endif
#ifndef CONV_3D_WINOGRAD_S8_W3_64X32R_TEX
#define CONV_3D_WINOGRAD_S8_W3_64X32R_TEX

template<int FH>
inline void conv3D_Winograd_s8_W3_64x32R_tex(jlong* streams, int &index, int length,
	const float*  X, cudaTextureObject_t texX, int IH, int IW,
	const float* CW,//[FH, FW, IC, OC]
	      float*  Y, int OH, int OW,
	int N, int IC, int OC,
	int ph, int pw,
	int GN, int GM, int &GNr, int &GMr)
{
	//======[Stage1 (OW % 6): Winograd_F(6, 3)]=========================================================================
	next_cudaStream(stream, streams, index, length);
	if ((pw <= 1) && (OW - IW - pw + 1 <= 0)) {//pw <= 1
#ifdef ENBALE_CONV3D_WINOGRAD_F5X4R_CHANNEL_TEMPLATE
		//OC = 2^x
		if      (OC ==   64) conv3dWinograd_f6x3_k64x192R_p1_OCT_tex(stream, texX, IH, IW, CW, FH, Y, OH, OW, N, IC,   64, ph, pw);
		else if (OC ==  128) conv3dWinograd_f6x3_k64x192R_p1_OCT_tex(stream, texX, IH, IW, CW, FH, Y, OH, OW, N, IC,  128, ph, pw);
		else if (OC ==  256) conv3dWinograd_f6x3_k64x192R_p1_OCT_tex(stream, texX, IH, IW, CW, FH, Y, OH, OW, N, IC,  256, ph, pw);
		else if (OC ==  512) conv3dWinograd_f6x3_k64x192R_p1_OCT_tex(stream, texX, IH, IW, CW, FH, Y, OH, OW, N, IC,  512, ph, pw);
		else if (OC == 1024) conv3dWinograd_f6x3_k64x192R_p1_OCT_tex(stream, texX, IH, IW, CW, FH, Y, OH, OW, N, IC, 1024, ph, pw);
		//OC = 64x
		else if (OC ==  192) conv3dWinograd_f6x3_k64x192R_p1_OCT_tex(stream, texX, IH, IW, CW, FH, Y, OH, OW, N, IC,  192, ph, pw);
		else if (OC ==  320) conv3dWinograd_f6x3_k64x192R_p1_OCT_tex(stream, texX, IH, IW, CW, FH, Y, OH, OW, N, IC,  320, ph, pw);
		else if (OC ==  384) conv3dWinograd_f6x3_k64x192R_p1_OCT_tex(stream, texX, IH, IW, CW, FH, Y, OH, OW, N, IC,  384, ph, pw);
		else if (OC ==  448) conv3dWinograd_f6x3_k64x192R_p1_OCT_tex(stream, texX, IH, IW, CW, FH, Y, OH, OW, N, IC,  448, ph, pw);
		else 
#endif
		conv3dWinograd_f6x3_k64x192R_p1_tex(stream, texX, IH, IW, CW, FH, Y, OH, OW, N, IC, OC, ph, pw);//common
	}
	else conv3dWinograd_f6x3_k64x192R_tex(stream, texX, IH, IW, CW, FH, Y, OH, OW, N, IC, OC, ph, pw);//common

	//======[Stage2 (OW % 4): Winograd_F(2+2, 3)]=======================================================================
	int OWr = OW % 6;
	if ((OWr >= 4) && !((N*OH) & 31)) {//Remainder: 4, 5
		next_cudaStream(stream1, streams, index, length);
		conv3dWinograd_f2x3_k64x128R4C_tex(stream1, (OW - OWr), texX, IH, IW, CW, FH, Y, OH, OW, N, IC, OC, ph, pw, OWr);
		OWr = OWr & 3;//OWr % 4
	}

	//======[Stage3 (OW % 2): Winograd_F(2, 3)]=========================================================================
	if ((OWr >= 2) && !((N*OH) & 63)) {//Remainder: 2, 3
		next_cudaStream(stream2, streams, index, length);
		conv3dWinograd_f2x3_k64x128RC_tex(stream2, (OW - OWr), texX, IH, IW, CW, FH, Y, OH, OW, N, IC, OC, ph, pw, OWr);
		OWr = OWr & 1;//OWr % 2
	}

	//======[Stage4: GEMM]==============================================================================================
	if (OWr > 0) {//Remainder: 1
		const int GN64 = ((OC  ) >> 6 << 6);      //GN   = 64x
		const int GM32 = ((N*OH) >> 5 << 5) * OWr;//N*OH = 32x
		index = 0;//save L2 cache: texX ! = X
		conv3D_Gemm_32x32RC(streams, index, length,
			X, IH, IW, CW, FH, 3, Y, OH, OW, OWr, IC, OC, 1, 1, ph, pw,
			GN64, GM32, 0, 0, OW - OWr);
	}
	
	//=====[Process GNr and GMr]========================================================================================
	GNr = GN & 63;//64 channels
	GMr = ((N*OH) & 31)*OW;//32 groups
}

#endif

#endif


//FW = 4
#ifndef CONV_3D_WINOGRAD_S8_W4_64X32R_TEXTURE
#define CONV_3D_WINOGRAD_S8_W4_64X32R_TEXTURE

#ifndef CONV_3D_WINOGRAD_S8_W4_64X32R_TEXTURE_BENCK_MARK_RTX3060TI
#define CONV_3D_WINOGRAD_S8_W4_64X32R_TEXTURE_BENCK_MARK_RTX3060TI

//[0] for: Feature = (160, 160), [N, IC, OC] = [32, 64, 64]
//LB = 4: Size = 50, Time = 5.75545 msec, Performace = 18656.1 GFlop/s
//WB = 4: Size = 50, Time = 5.3804  msec, Performace = 19956.5 GFlop/s
//cuDNN-NCHW-implicit-prec: Size = 50, Time = 8.397 msec, Performance = 12787.2 GFlop/s
//cuDNN-NHWC-implicit-prec: Size = 50, Time = 7.414 msec, Performance = 14482.6 GFlop/s
//cuDNN-NHWC-GEMM-implicit: Size = 50, Time = 33.514 msec, Performance = 3203.8 GFlop/s

//[1] for: Feature = (128, 128), [N, IC, OC] = [32, 64, 64]
//LB = 4: Size = 32, Time = 3.85289 msec, Performace = 17835.8 GFlop/s
//WB = 4: Size = 32, Time = 3.60314 msec, Performace = 19072.1 GFlop/s
//cuDNN-NCHW-implicit-prec: Size = 32, Time = 5.373 msec, Performance = 12789.8 GFlop/s
//cuDNN-NHWC-implicit-prec: Size = 32, Time = 4.762 msec, Performance = 14430.8 GFlop/s
//cuDNN-NHWC-GEMM-implicit: Size = 32, Time = 21.375 msec, Performance = 3214.9 GFlop/s

//[2] for: Feature = (80, 80), [N, IC, OC] = [128, 64, 64]
//LB = 4: Size = 50, Time = 5.75289 msec, Performace = 18664.4 GFlop/s
//WB = 4: Size = 50, Time = 5.37185 msec, Performace = 19988.3 GFlop/s
//cuDNN-NCHW-implicit-prec: Size = 50, Time = 8.339 msec, Performance = 12876.1 GFlop/s
//cuDNN-NHWC-implicit-prec: Size = 50, Time = 7.437 msec, Performance = 14437.8 GFlop/s
//cuDNN-NHWC-GEMM-implicit: Size = 50, Time = 33.32 msec, Performance = 3222.5 GFlop/s

//[3] for: Feature = (64, 64), [N, IC, OC] = [128, 64, 64]
//LB = 4: Size = 32, Time = 4.01459 msec, Performace = 17117.4 GFlop/s
//WB = 4: Size = 32, Time = 3.72449 msec, Performace = 18450.7 GFlop/s
//cuDNN-NCHW-implicit-prec: Size = 32, Time = 5.392 msec, Performance = 12744.7 GFlop/s
//cuDNN-NHWC-implicit-prec: Size = 32, Time = 4.775 msec, Performance = 14391.5 GFlop/s
//cuDNN-NHWC-GEMM-implicit: Size = 32.00, Time = 21.33 msec, Performance = 3221.7 GFlop/s

//[4] for: Feature = (40, 40), [N, IC, OC] = [128, 128, 128]
//LB = 4: Size = 50, Time = 5.50282 msec, Performace = 19512.6 GFlop/s
//WB = 4: Size = 50, Time = 5.13534 msec, Performace = 20908.9 GFlop/s
//cuDNN-NCHW-implicit-prec: Size = 50, Time = 6.738 msec, Performance = 15935.6 GFlop/s
//cuDNN-NHWC-implicit-prec: Size = 50, Time = 6.961 msec, Performance = 15425.1 GFlop/s
//cuDNN-NHWC-GEMM-implicit: Size = 50, Time = 23.7 msec, Performance = 4530.6 GFlop/s

//[5] for: Feature = (32, 32), [N, IC, OC] = [128, 128, 128]
//LB = 4: Size = 32, Time = 3.75329 msec, Performace = 18309.2 GFlop/s
//WB = 4: Size = 32, Time = 3.52311 msec, Performace = 19505.3 GFlop/s
//cuDNN-NCHW-implicit-prec: Size = 32, Time = 4.585 msec, Performance = 14987.9 GFlop/s
//cuDNN-NHWC-implicit-prec: Size = 32, Time = 4.463 msec, Performance = 15397.6 GFlop/s
//cuDNN-NHWC-GEMM-implicit: Size = 3, Time = 15.19 msec, Performance = 4524.0 GFlop/s

//[6] for: Feature = (20, 20), [N, IC, OC] = [128, 256, 256]
//LB = 4: Size = 50, Time = 5.32529 msec, Performace = 20163.1 GFlop/s
//WB = 4: Size = 50, Time = 4.9916  msec, Performace = 21510.9 GFlop/s
//cuDNN-NCHW-implicit-prec: Size = 50, Time = 6.565 msec, Performance = 16355.5 GFlop/s
//cuDNN-NHWC-implicit-prec: Size = 50, Time = 6.482 msec, Performance = 16564.9 GFlop/s
//cuDNN-NHWC-GEMM-implicit: Size = 50, Time = 17.35 msec, Performance = 6188.7 GFlop/s

//[7] for: Feature = (16, 16), [N, IC, OC] = [128, 256, 256]
//LB = 4: Size = 32, Time = 3.75933 msec, Performace = 18279.7 GFlop/s
//WB = 4: Size = 32, Time = 3.50263 msec, Performace = 19619.4 GFlop/s
//cuDNN-NCHW-implicit-prec: Size = 32, Time = 4.206 msec, Performance = 16338.4 GFlop/s
//cuDNN-NHWC-implicit-prec: Size = 32, Time = 4.193 msec, Performance = 16389.1 GFlop/s
//cuDNN-NHWC-GEMM-implicit: Size = 32, Time = 11.055 msec, Performance = 6216.1 GFlop/s

//[8] for: Feature = (10, 10), [N, IC, OC] = [128, 512, 512]
//LB = 4: Size = 50, Time = 5.37366 msec, Performace = 19981.6 GFlop/s
//WB = 4: Size = 50, Time = 4.91173 msec, Performace = 21860.8 GFlop/s
//cuDNN-NCHW-implicit-prec: Size = 50, Time = 6.543 msec, Performance = 16410.5 GFlop/s
//cuDNN-NHWC-implicit-prec: Size = 50, Time = 6.407 msec, Performance = 16758.9 GFlop/s
//cuDNN-NHWC-GEMM-implicit: Size = 50, Time = 17.14 msec, Performance = 6264.5 GFlop/s

//[9] for: Feature = (8, 8), [N, IC, OC] = [128, 512, 512]
//LB = 4: Size = 32, Time = 4.45407 msec, Performace = 15428.5 GFlop/s
//WB = 4: Size = 32, Time = 4.10901 msec, Performace = 16724.1 GFlop/s
//cuDNN-NCHW-implicit-prec: Size = 32, Time = 4.214 msec, Performance = 16307.4 GFlop/s
//cuDNN-NHWC-implicit-prec: Size = 32, Time = 4.138 msec, Performance = 16606.9 GFlop/s
//cuDNN-NHWC-GEMM-implicit: Size = 32, Time = 10.81 msec, Performance = 6357.0 GFlop/s

#endif

//(1) FW = 4
//(2) sh = sw = 1
//(3) GN: (OC    ) % 64
//(4) GM: (N * OH) % 32
//(5) IC % 8 == 0
//(6) OW >= 5
#ifndef COMPILE
#define CONV_3D_WINOGRAD_S8_W4_64X32R_TEX
#endif
#ifndef CONV_3D_WINOGRAD_S8_W4_64X32R_TEX
#define CONV_3D_WINOGRAD_S8_W4_64X32R_TEX

template<int FH>
inline void conv3D_Winograd_s8_W4_64x32R_tex(jlong* streams, int &index, int length,
	const float*  X, cudaTextureObject_t texX, int IH, int IW,
	const float* CW,//[FH, FW, IC, OC]
	      float*  Y, int OH, int OW,
	int N, int IC, int OC,
	int ph, int pw,
	int GN, int GM, int &GNr, int &GMr)
{
	///======[Stage1 (OW % 5): Winograd_F(5, 4)]=========================================================================
	next_cudaStream(stream, streams, index, length);
	if ((pw <= 2) && (OW - IW - pw + 1 <= 0)) {//pw <= 2
#ifdef ENBALE_CONV3D_WINOGRAD_F5X4R_CHANNEL_TEMPLATE
		//OC = 2^x
		if      (OC ==   64) conv3dWinograd_f5x4_k64x160R_p2_OCT_tex(stream, texX, IH, IW, CW, FH, Y, OH, OW, N, IC,   64, ph, pw);
		else if (OC ==  128) conv3dWinograd_f5x4_k64x160R_p2_OCT_tex(stream, texX, IH, IW, CW, FH, Y, OH, OW, N, IC,  128, ph, pw);
		else if (OC ==  256) conv3dWinograd_f5x4_k64x160R_p2_OCT_tex(stream, texX, IH, IW, CW, FH, Y, OH, OW, N, IC,  256, ph, pw);
		else if (OC ==  512) conv3dWinograd_f5x4_k64x160R_p2_OCT_tex(stream, texX, IH, IW, CW, FH, Y, OH, OW, N, IC,  512, ph, pw);
		else if (OC == 1024) conv3dWinograd_f5x4_k64x160R_p2_OCT_tex(stream, texX, IH, IW, CW, FH, Y, OH, OW, N, IC, 1024, ph, pw);
		//OC = 64x
		else if (OC ==  192) conv3dWinograd_f5x4_k64x160R_p2_OCT_tex(stream, texX, IH, IW, CW, FH, Y, OH, OW, N, IC,  192, ph, pw);
		else if (OC ==  320) conv3dWinograd_f5x4_k64x160R_p2_OCT_tex(stream, texX, IH, IW, CW, FH, Y, OH, OW, N, IC,  320, ph, pw);
		else if (OC ==  384) conv3dWinograd_f5x4_k64x160R_p2_OCT_tex(stream, texX, IH, IW, CW, FH, Y, OH, OW, N, IC,  384, ph, pw);
		else if (OC ==  448) conv3dWinograd_f5x4_k64x160R_p2_OCT_tex(stream, texX, IH, IW, CW, FH, Y, OH, OW, N, IC,  448, ph, pw); 
		else 
#endif
		conv3dWinograd_f5x4_k64x160R_p2_tex(stream, texX, IH, IW, CW, FH, Y, OH, OW, N, IC, OC, ph, pw);
	}
	else conv3dWinograd_f5x4_k64x160R_tex(stream, texX, IH, IW, CW, FH, Y, OH, OW, N, IC, OC, ph, pw);//common

	//======[Stage2 (OW % 3): Winograd_F(3, 2)]=========================================================================
	int OWr = OW % 5;
	if ((OWr >= 3) && !((N*OH) & 63)) {//Remainder: 3, 4
		conv3dWinograd_SFW_f3x2_k64x192RC_tex(stream, (OW - OWr), texX, IH, IW, CW, FH, 4, Y, OH, OW, N, IC, OC, ph, pw, OWr);
		OWr = OWr % 3;//OWr % 3
	}

	//======[Stage3: GEMM]==============================================================================================
	if (OWr > 0) {//Remainder: 1, 2
		const int GN64 = ((OC  ) >> 6 << 6);      //GN   = 64x
		const int GM32 = ((N*OH) >> 5 << 5) * OWr;//N*OH = 32x
		index = 0;//save L2 cache: texX != X
		conv3D_Gemm_32x32RC(streams, index, length,
			X, IH, IW, CW, FH, 4, Y, OH, OW, OWr, IC, OC, 1, 1, ph, pw,
			GN64, GM32, 0, 0, OW - OWr);
	}
	
	//=====[Process GNr and GMr]========================================================================================
	GNr = GN & 63;           //64 channels
	GMr = ((N*OH) & 31) * OW;//32 groups
}

#endif

#endif


//FW = 5
#ifndef CONV_3D_WINOGRAD_S8_W5_64X32R_TEXTURE
#define CONV_3D_WINOGRAD_S8_W5_64X32R_TEXTURE

#ifndef CONV_3D_WINOGRAD_S8_W5_64X32R_TEXTURE_BENCK_MARK_RTX3060TI
#define CONV_3D_WINOGRAD_S8_W5_64X32R_TEXTURE_BENCK_MARK_RTX3060TI

//[0] for: Feature = (128, 128), [N, IC, OC] = [32, 64, 64]
//LB = 4: Size = 50, Time = 5.72897 msec, Performace = 18742.3 GFlop/s
//WB = 4: Size = 50, Time = 5.27933 msec, Performace = 20338.6 GFlop/s
//cuDNN-NCHW-GEMM-implicit-prec: Size = 50, Time = 7.603 msec, Performance = 14122.6 GFlop/s
//cuDNN-NHWC-GEMM-implicit-prec: Size = 50, Time = 6.889 msec, Performance = 15586.3 GFlop/s
//cuDNN-NHWC-GEMM-implicit: Size = 50, Time = 33.3 msec, Performance = 3223 GFlop/s

//[1] for: Feature = (66, 66), [N, IC, OC] = [128, 64, 64]
//LB = 4: Size = 53.173828, Time = 6.44339 msec, Performace = 17722 GFlop/s
//WB = 4: Size = 53.173828, Time = 6.09289 msec, Performace = 18741.5 GFlop/s
//cuDNN-NCHW-GEMM-implicit-prec: Size = 53.173828, Time = 6.896 msec, Performance = 16558.9 GFlop/s
//cuDNN-NHWC-GEMM-implicit-prec: Size = 53.173828, Time = 6.897 msec, Performance = 16556.5 GFlop/s
//cuDNN-NHWC-GEMM-implicit: Size = 53.173828, Time = 35.06 msec, Performance = 3257 GFlop/s

//[2] for: Feature = (64, 64), [N, IC, OC] = [32, 128, 128]
//LB = 4: Size = 50, Time = 5.74325 msec, Performace = 18695.7 GFlop/s
//WB = 4: Size = 50, Time = 5.27971 msec, Performace = 20337.1 GFlop/s
//cuDNN-NCHW-GEMM-implicit-prec: Size = 50, Time = 6.826 msec, Performance = 15730.2 GFlop/s
//cuDNN-NHWC-GEMM-implicit-prec: Size = 50, Time = 6.466 msec, Performance = 16605.9 GFlop/s
//cuDNN-NHWC-GEMM-implicit: Size = 50, Time = 33.1 msec, Performance = 3243 GFlop/s

//[3]  for: Feature = (48, 48), [N, IC, OC] = [128, 128, 128]
//LB = 4: Size = 112.5, Time = 12.5088 msec, Performace = 19313.7 GFlop/s
//WB = 4: Size = 112.5, Time = 11.7694 msec, Performace = 20527.1 GFlop/s
//cuDNN-NCHW-GEMM-implicit-prec: Size = 112.5, Time = 14.333 msec, Performance = 16855.6 GFlop/s
//cuDNN-NHWC-GEMM-implicit-prec: Size = 112.5, Time = 14.416 msec, Performance = 16758.6 GFlop/s
//cuDNN-NHWC-GEMM-implicit: Size = 112.5, Time = 38.65 msec, Performance = 6250.8 GFlop/s

//[4] for: Feature = (34, 34), [N, IC, OC] = [128, 128, 128]
//LB = 4: Size = 56.445313, Time = 6.62145 msec, Performace = 18306.5 GFlop/s
//WB = 4: Size = 56.445313, Time = 6.25877 msec, Performace = 19367.3 GFlop/s
//cuDNN-NCHW-GEMM-implicit-prec: Size = 56.445313, Time = 7.410 msec, Performance = 16358.4 GFlop/s
//cuDNN-NHWC-GEMM-implicit-prec: Size = 56.445313, Time = 7.688 msec, Performance = 15766.8 GFlop/s
//cuDNN-NHWC-GEMM-implicit: Size = 56.445313, Time = 19.405 msec, Performance = 6246.6 GFlop/s

//[5] for: Feature = (32, 32), [N, IC, OC] = [128, 128, 128]
//LB = 4: Size = 50, Time = 5.50963 msec, Performace = 19488.4 GFlop/s
//WB = 4: Size = 50, Time = 5.12471 msec, Performace = 20952.2 GFlop/s
//cuDNN-NCHW-GEMM-implicit-prec: Size = 50, Time = 6.891 msec, Performance = 15581.8 GFlop/s
//cuDNN-NHWC-GEMM-implicit-prec: Size = 50, Time = 6.788 msec, Performance = 15818.2 GFlop/s
//cuDNN-NHWC-GEMM-implicit: Size = 50, Time = 17.245 msec, Performance = 6226.4 GFlop / s

//[6] for: Feature = (18, 18), [N, IC, OC] = [128, 256, 256]
//LB = 4: Size = 63.28125, Time = 7.47727 msec, Performace = 18174.5 GFlop/s
//WB = 4: Size = 63.28125, Time = 7.00282 msec, Performace = 19405.8 GFlop/s
//cuDNN-NCHW-GEMM-implicit-prec: Size = 63.28125, Time = 8.241 msec, Performance = 16490.2 GFlop/s
//cuDNN-NHWC-GEMM-implicit-prec: Size = 63.28125, Time = 8.194 msec, Performance = 16584.8 GFlop/s
//cuDNN-NHWC-GEMM-implicit: Size = 63.28125, Time = 22.21 msec, Performance = 6118.7 GFlop/s

//[7] for: Feature = (16, 16), [N, IC, OC] = [128, 256, 256]
//LB = 4: Size = 50, Time = 5.37508 msec, Performace = 19976.3 GFlop/s
//WB = 4: Size = 50, Time = 4.9361  msec, Performace = 21752.8 GFlop/s
//cuDNN-NCHW-GEMM-implicit-prec: Size = 50, Time = 6.936 msec, Performance = 15480.7 GFlop/s
//cuDNN-NHWC-GEMM-implicit-prec: Size = 50, Time = 6.442 msec, Performance = 16667.8 GFlop/s
//cuDNN-NHWC-GEMM-implicit: Size = 50, Time = 17.225 msec, Performance = 6233.6 GFlop/s

//[8] for: Feature = (10, 10), [N, IC, OC] = [128, 512, 512]
//LB = 4: Size = 78.125, Time = 10.7425 msec, Performace = 15617.6 GFlop/s
//WB = 4: Size = 78.125, Time = 9.98231 msec, Performace = 16806.9 GFlop/s
//cuDNN-NCHW-GEMM-implicit-prec: Size = 78.125, Time = 10.027 msec, Performance = 16732.0 GFlop/s
//cuDNN-NHWC-GEMM-implicit-prec: Size = 78.125, Time =  9.908 msec, Performance = 16933.0 GFlop/s
//cuDNN-NHWC-GEMM-implicit: Size = 78.125, Time = 29.15 msec, Performance = 5755.5 GFlop/s

//[9] for: Feature = (8, 8), [N, IC, OC] = [128, 512, 512]
//LB = 4: Size = 50, Time = 5.59623 msec, Performace = 19186.9 GFlop/s
//WB = 4: Size = 50, Time = 5.07147 msec, Performace = 21172.2 GFlop/s
//cuDNN-NCHW-GEMM-implicit-prec: Size = 50, Time = 6.506 msec, Performance = 16503.9 GFlop/s
//cuDNN-NHWC-GEMM-implicit-prec: Size = 50, Time = 6.254 msec, Performance = 17168.8 GFlop/s
//cuDNN-NHWC-GEMM-implicit: Size = 50, Time = 16.93 msec, Performance = 6342.2 GFlop/s

#endif

#ifndef CONV_3D_WINOGRAD_S8_W5_RUSE_64X32R_TEXTURE_BENCK_MARK_RTX3060TI
#define CONV_3D_WINOGRAD_S8_W5_RUSE_64X32R_TEXTURE_BENCK_MARK_RTX3060TI

//[0] for: Feature = (128, 128), [N, IC, OC] = [32, 64, 64]
//LB = 4: Size = 50, Time = 5.01505 msec, Performace = 21410.4 GFlop/s
//WB = 4: Size = 50, Time = 4.54375 msec, Performace = 23631.2 GFlop/s
//cuDNN-NCHW-GEMM-implicit-prec: Size = 50, Time = 7.603 msec, Performance = 14122.6 GFlop/s
//cuDNN-NHWC-GEMM-implicit-prec: Size = 50, Time = 6.889 msec, Performance = 15586.3 GFlop/s
//cuDNN-NHWC-GEMM-implicit: Size = 50, Time = 33.3 msec, Performance = 3223 GFlop/s

//[1] for: Feature = (66, 66), [N, IC, OC] = [128, 64, 64]
//LB = 4: Size = 53.173828, Time = 5.71977 msec, Performace = 19964.1 GFlop/s
//WB = 4: Size = 53.173828, Time = 5.18733 msec, Performace = 22013.2 GFlop/
//cuDNN-NCHW-GEMM-implicit-prec: Size = 53.173828, Time = 6.896 msec, Performance = 16558.9 GFlop/s
//cuDNN-NHWC-GEMM-implicit-prec: Size = 53.173828, Time = 6.897 msec, Performance = 16556.5 GFlop/s
//cuDNN-NHWC-GEMM-implicit: Size = 53.173828, Time = 35.06 msec, Performance = 3257 GFlop/s

//[2] for: Feature = (64, 64), [N, IC, OC] = [32, 128, 128]
//LB = 4: Size = 50, Time = 5.11399 msec, Performace = 20996.2 GFlop/s
//WB = 4: Size = 50, Time = 4.643   msec, Performace = 23126   GFlop/s
//cuDNN-NCHW-GEMM-implicit-prec: Size = 50, Time = 6.826 msec, Performance = 15730.2 GFlop/s
//cuDNN-NHWC-GEMM-implicit-prec: Size = 50, Time = 6.466 msec, Performance = 16605.9 GFlop/s
//cuDNN-NHWC-GEMM-implicit: Size = 50, Time = 33.1 msec, Performance = 3243 GFlop/s

//[3]  for: Feature = (48, 48), [N, IC, OC] = [128, 128, 128]
//LB = 4: Size = 112.5, Time = 11.6415 msec, Performace = 20752.7 GFlop/s
//WB = 4: Size = 112.5, Time = 10.5723 msec, Performace = 22851.3 GFlop/s
//cuDNN-NCHW-GEMM-implicit-prec: Size = 112.5, Time = 14.333 msec, Performance = 16855.6 GFlop/s
//cuDNN-NHWC-GEMM-implicit-prec: Size = 112.5, Time = 14.416 msec, Performance = 16758.6 GFlop/s
//cuDNN-NHWC-GEMM-implicit: Size = 112.5, Time = 38.65 msec, Performance = 6250.8 GFlop/s

//[4] for: Feature = (34, 34), [N, IC, OC] = [128, 128, 128]
//LB = 4: Size = 56.445313, Time = 6.32294 msec, Performace = 19170.7 GFlop/s
//WB = 4: Size = 56.445313, Time = 5.7957  msec, Performace = 20914.7 GFlop/s
//cuDNN-NCHW-GEMM-implicit-prec: Size = 56.445313, Time = 7.410 msec, Performance = 16358.4 GFlop/s
//cuDNN-NHWC-GEMM-implicit-prec: Size = 56.445313, Time = 7.688 msec, Performance = 15766.8 GFlop/s
//cuDNN-NHWC-GEMM-implicit: Size = 56.445313, Time = 19.405 msec, Performance = 6246.6 GFlop/s

//[5] for: Feature = (32, 32), [N, IC, OC] = [128, 128, 128]
//LB = 4: Size = 50, Time = 5.05927 msec, Performace = 21223.2 GFlop/s
//WB = 4: Size = 50, Time = 4.6013  msec, Performace = 23335.6 GFlop/s
//cuDNN-NCHW-GEMM-implicit-prec: Size = 50, Time = 6.891 msec, Performance = 15581.8 GFlop/s
//cuDNN-NHWC-GEMM-implicit-prec: Size = 50, Time = 6.788 msec, Performance = 15818.2 GFlop/s
//cuDNN-NHWC-GEMM-implicit: Size = 50, Time = 17.245 msec, Performance = 6226.4 GFlop / s

//[6] for: Feature = (18, 18), [N, IC, OC] = [128, 256, 256]
//LB = 4: Size = 63.28125, Time = 7.05993 msec, Performace = 19248.8 GFlop/s
//WB = 4: Size = 63.28125, Time = 6.35183 msec, Performace = 21394.7 GFlop/s
//cuDNN-NCHW-GEMM-implicit-prec: Size = 63.28125, Time = 8.241 msec, Performance = 16490.2 GFlop/s
//cuDNN-NHWC-GEMM-implicit-prec: Size = 63.28125, Time = 8.194 msec, Performance = 16584.8 GFlop/s
//cuDNN-NHWC-GEMM-implicit: Size = 63.28125, Time = 22.21 msec, Performance = 6118.7 GFlop/s

//[7] for: Feature = (16, 16), [N, IC, OC] = [128, 256, 256]
//LB = 4: Size = 50, Time = 4.93715 msec, Performace = 21748.2 GFlop/s
//WB = 4: Size = 50, Time = 4.38719 msec, Performace = 24474.5 GFlop/s
//cuDNN-NCHW-GEMM-implicit-prec: Size = 50, Time = 6.936 msec, Performance = 15480.7 GFlop/s
//cuDNN-NHWC-GEMM-implicit-prec: Size = 50, Time = 6.442 msec, Performance = 16667.8 GFlop/s
//cuDNN-NHWC-GEMM-implicit: Size = 50, Time = 17.225 msec, Performance = 6233.6 GFlop/s

//[8] for: Feature = (10, 10), [N, IC, OC] = [128, 512, 512]
//LB = 4: Size = 78.125, Time = 10.0222 msec, Performace = 16740.1 GFlop/s
//WB = 4: Size = 78.125, Time = 9.11784 msec, Performace = 18400.4 GFlop/s
//cuDNN-NCHW-GEMM-implicit-prec: Size = 78.125, Time = 10.027 msec, Performance = 16732.0 GFlop/s
//cuDNN-NHWC-GEMM-implicit-prec: Size = 78.125, Time =  9.908 msec, Performance = 16933.0 GFlop/s
//cuDNN-NHWC-GEMM-implicit: Size = 78.125, Time = 29.15 msec, Performance = 5755.5 GFlop/s

//[9] for: Feature = (8, 8), [N, IC, OC] = [128, 512, 512]
//LB = 4: Size = 50, Time = 4.91652 msec, Performace = 21839.5 GFlop/s
//WB = 4: Size = 50, Time = 4.32837 msec, Performace = 24807.1 GFlop/s
//cuDNN-NCHW-GEMM-implicit-prec: Size = 50, Time = 6.506 msec, Performance = 16503.9 GFlop/s
//cuDNN-NHWC-GEMM-implicit-prec: Size = 50, Time = 6.254 msec, Performance = 17168.8 GFlop/s
//cuDNN-NHWC-GEMM-implicit: Size = 50, Time = 16.93 msec, Performance = 6342.2 GFlop/s

#endif

//(1) FW = 5
//(2) sh = sw = 1
//(3) GN: (OC    ) % 64 == 0
//(4) GM: (N * OH) % 32 == 0
//(5) IC % 8 == 0
//(6) OW >= 4
#ifndef COMPILE
#define CONV_3D_WINOGRAD_S8_W5_64X32R_TEX
#endif
#ifndef CONV_3D_WINOGRAD_S8_W5_64X32R_TEX
#define CONV_3D_WINOGRAD_S8_W5_64X32R_TEX

template<int FH>
inline void conv3D_Winograd_s8_W5_64x32R_tex(jlong* streams, int &index, int length,
	const float*  X, cudaTextureObject_t texX, int IH, int IW,
	const float* CW,//[FH, FW, IC, OC]
	      float*  Y, int OH, int OW,
	int N, int IC, int OC,
	int ph, int pw,
	int GN, int GM, int &GNr, int &GMr)
{
	//======[Stage1 (OW % 4): Winograd_F(4, 5)]=========================================================================
	next_cudaStream(stream, streams, index, length);
	if ((pw <= 2) && (OW - IW - pw + 2 <= 0)) {//pw <= 2
		if (OW >= 8) {
			conv3dWinograd_f4x5_ruse_k64x128R_p2_tex(stream, texX, IH, IW, CW, FH, Y, OH, OW, N, IC, OC, ph, pw);
			const int OWr = OW & 7;//align: OW % 4
			if (OWr >= 4) conv3dWinograd_f4x5_k64x128RC_p2_tex(stream, (OW - OWr), texX, IH, IW, CW, FH, Y, OH, OW, N, IC, OC, ph, pw, OWr);
		}
		else conv3dWinograd_f4x5_k64x128R_p2_tex(stream, texX, IH, IW, CW, FH, Y, OH, OW, N, IC, OC, ph, pw);
	}
	else conv3dWinograd_f4x5_k64x128R_tex(stream, texX, IH, IW, CW, FH, Y, OH, OW, N, IC, OC, ph, pw);//common

	//======[Stage2: GEMM]==============================================================================================
	const int OWr = OW & 3;
	if (OWr > 0) {//Remainder: 1, 2, 3
		const int GN64 = ((OC  ) >> 6 << 6);      //GN   = 64x
		const int GM32 = ((N*OH) >> 5 << 5) * OWr;//N*OH = 32x
		index = 0;//save L2 cache: texX != X
		conv3D_Gemm_32x32RC(streams, index, length,
			X, IH, IW, CW, FH, 5, Y, OH, OW, OWr, IC, OC, 1, 1, ph, pw,
			GN64, GM32, 0, 0, OW - OWr);
	}

	//=====[Process GNr and GMr]========================================================================================
	GNr = GN & 63;//64 channels
	GMr = ((N*OH) & 31) * OW;//32 groups
}

#endif

#endif


//FW = 6
#ifndef CONV_3D_WINOGRAD_S8_W6_64X32R_TEXTURE
#define CONV_3D_WINOGRAD_S8_W6_64X32R_TEXTURE

#ifndef CONV_3D_WINOGRAD_S8_W6_64X32R_TEXTURE_BENCK_MARK_RTX3060TI
#define CONV_3D_WINOGRAD_S8_W6_64X32R_TEXTURE_BENCK_MARK_RTX3060TI

//[0] for: Feature = (128, 128), [N, IC, OC] = [32, 64, 64]
//LB = 4: Size = 72, Time = 9.11207 msec, Performace = 16968.6 GFlop/s
//WB = 4: Size = 72, Time = 8.55456 msec, Performace = 18074.4 GFlop/s
//cuDNN-NCHW-implicit-prec: Size = 72, Time = 10.238 msec, Performance = 15102.4 GFlop/s
//cuDNN-NHWC-implicit-prec: Size = 72, Time =  9.483 msec, Performance = 16304.8 GFlop/s
//cuDNN-NHWC-GEMM-implicit: Size = 72, Time = 47.71 msec, Performance = 3240.8 GFlop/s

//[1]  for: Feature = (96, 96), [N, IC, OC] = [32, 64, 64]
//LB = 4: Size = 40.5, Time = 5.0338  msec, Performace = 17277.8 GFlop/s
//WB = 4: Size = 40.5, Time = 4.74598 msec, Performace = 18325.6 GFlop/s
//cuDNN-NCHW-implicit-prec: Size = 40.5, Time = 5.778 msec, Performance = 15052.5 GFlop/s
//cuDNN-NHWC-implicit-prec: Size = 40.5, Time = 5.401 msec, Performance = 16103.1 GFlop/s
//cuDNN-NHWC-GEMM-implicit: Size = 40.5, Time = 26.815 msec, Performance = 3243.4 GFlop/s

//[2] for: Feature = (64, 64), [N, IC, OC] = [128, 64, 64]
//LB = 4: Size = 72, Time = 9.19208 msec, Performace = 16820.9 GFlop/s
//WB = 4: Size = 72, Time = 8.62389 msec, Performace = 17929.1 GFlop/s
//cuDNN-NCHW-implicit-prec: Size = 72, Time = 10.092 msec, Performance = 15320.9 GFlop/s
//cuDNN-NHWC-implicit-prec: Size = 72, Time =  9.502 msec, Performance = 16272.3 GFlop/s
//cuDNN-NHWC-GEMM-implicit: Size = 72, Time = 47.29 msec, Performance = 3269.6 GFlop/s

//[3] for: Feature = (48, 48), [N, IC, OC] = [128, 64, 64]
//LB = 4: Size = 40.5, Time = 5.05368 msec, Performace = 17209.9 GFlop/s
//WB = 4: Size = 40.5, Time = 4.74442 msec, Performace = 18331.6 GFlop/s
//cuDNN-NCHW-implicit-prec: Size = 40.5, Time = 5.758 msec, Performance = 15104.7 GFlop/s
//cuDNN-NHWC-implicit-prec: Size = 40.5, Time = 5.395 msec, Performance = 16121.1 GFlop/s
//cuDNN-NHWC-GEMM-implicit: Size = 40.5, Time = 26.525 msec, Performance = 3278.9 GFlop/s

//[4] for: Feature = (32, 32), [N, IC, OC] = [128, 128, 128]
//LB = 4: Size = 72, Time = 9.18944 msec, Performace = 16825.7 GFlop/s
//WB = 4: Size = 72, Time = 8.58351 msec, Performace = 18013.5 GFlop/s
//cuDNN-NCHW-implicit-prec: Size = 72, Time = 9.379 msec, Performance = 16485.6 GFlop/s
//cuDNN-NHWC-implicit-prec: Size = 72, Time = 9.505 msec, Performance = 16267.1 GFlop/s
//cuDNN-NHWC-GEMM-implicit: Size = 72, Time = 24.62 msec, Performance = 6280.2 GFlop/s

//[5] for: Feature = (24, 24), [N, IC, OC] = [128, 128, 128]
//LB = 4: Size = 40.5, Time = 4.9814 msec, Performace = 17459.4 GFlop/s
//WB = 4: Size = 40.5, Time = 4.6828 msec, Performace = 18572.9 GFlop/s
//cuDNN-NCHW-implicit-prec: Size = 40.5, Time = 5.423 msec, Performance = 16037.8 GFlop/s
//cuDNN-NHWC-implicit-prec: Size = 40.5, Time = 5.253 msec, Performance = 16556.8 GFlop/s
//cuDNN-NHWC-GEMM-implicit: Size = 40.5, Time = 13.85 msec, Performance = 6279.6 GFlop/s

//[6] for: Feature = (16, 16), [N, IC, OC] = [128, 256, 256]
//LB = 4: Size = 72, Time = 9.18034 msec, Performace = 16842.4 GFlop/s
//WB = 4: Size = 72, Time = 8.57575 msec, Performace = 18029.8 GFlop/s
//cuDNN-NCHW-implicit-prec: Size = 72, Time = 9.003 msec, Performance = 17174.1 GFlop/s
//cuDNN-NHWC-implicit-prec: Size = 72, Time = 9.149 msec, Performance = 16900.1 GFlop/s
//cuDNN-NHWC-GEMM-implicit: Size = 72, Time = 24.935 msec, Performance = 6200.9 GFlop/s

//[7] for: Feature = (12, 12), [N, IC, OC] = [128, 256, 256]
//LB = 4: Size = 40.5, Time = 4.98015 msec, Performace = 17463.9 GFlop/s
//WB = 4: Size = 40.5, Time = 4.62677 msec, Performace = 18797.8 GFlop/s
//cuDNN-NCHW-implicit-prec: Size = 40.5, Time = 5.352 msec, Performance = 16250.6 GFlop/s
//cuDNN-NHWC-implicit-prec: Size = 40.5, Time = 5.327 msec, Performance = 16326.8 GFlop/s
//cuDNN-NHWC-GEMM-implicit: Size = 40.5, Time = 13.76 msec, Performance = 6320.7 GFlop/s

//[8] for: Feature = (8, 8), [N, IC, OC] = [128, 512, 512]
//LB = 4: Size = 72, Time = 10.2182 msec, Performace = 15131.8 GFlop/s
//WB = 4: Size = 72, Time =  9.1880 msec, Performace = 16828.4 GFlop/s
//cuDNN-NCHW-implicit-prec: Size = 72, Time = 9.231 msec, Performance = 16750.0 GFlop/s
//cuDNN-NHWC-implicit-prec: Size = 72, Time = 9.174 msec, Performance = 16854.1 GFlop/s
//cuDNN-NHWC-GEMM-implicit: Size = 72, Time = 23.295 msec, Performance = 6637.4 GFlop/s

//[9] for: Feature = (6, 6), [N, IC, OC] = [128, 512, 512]
//WB = 4: Size = 40.5, Time = 5.29974 msec, Performace = 16410.8 GFlop/s
//LB = 4: Size = 40.5, Time = 4.75506 msec, Performace = 18290.6 GFlop/s
//cuDNN-NCHW-implicit-prec: Size = 40.5, Time = 5.439 msec, Performance = 15990.6 GFlop/s
//cuDNN-NHWC-implicit-prec: Size = 40.5, Time = 5.344 msec, Performance = 16274.9 GFlop/s
//cuDNN-NHWC-GEMM-implicit: Size = 40.5, Time = 13 msec, Performance = 6690.2 GFlop/s

#endif

#ifndef CONV_3D_WINOGRAD_S8_W6_RUSE_64X32R_TEXTURE_BENCK_MARK_RTX3060TI
#define CONV_3D_WINOGRAD_S8_W6_RUSE_64X32R_TEXTURE_BENCK_MARK_RTX3060TI

//[0] for: Feature = (128, 128), [N, IC, OC] = [32, 64, 64]
//LB = 4: Size = 72, Time = 8.11787 msec, Performace = 19290.9 GFlop/s
//WB = 4: Size = 72, Time = 7.19731 msec, Performace = 21482.9 GFlop/s
//cuDNN-NCHW-implicit-prec: Size = 72, Time = 10.238 msec, Performance = 15102.4 GFlop/s
//cuDNN-NHWC-implicit-prec: Size = 72, Time =  9.483 msec, Performance = 16304.8 GFlop/s
//cuDNN-NHWC-GEMM-implicit: Size = 72, Time = 47.71 msec, Performance = 3240.8 GFlop/s

//[1]  for: Feature = (96, 96), [N, IC, OC] = [32, 64, 64]
//LB = 4: Size = 40.5, Time = 4.4255  msec, Performace = 19652.7 GFlop/s
//WB = 4: Size = 40.5, Time = 3.99327 msec, Performace = 21779.9 GFlop/s
//cuDNN-NCHW-implicit-prec: Size = 40.5, Time = 5.778 msec, Performance = 15052.5 GFlop/s
//cuDNN-NHWC-implicit-prec: Size = 40.5, Time = 5.401 msec, Performance = 16103.1 GFlop/s
//cuDNN-NHWC-GEMM-implicit: Size = 40.5, Time = 26.815 msec, Performance = 3243.4 GFlop/s

//[2] for: Feature = (64, 64), [N, IC, OC] = [128, 64, 64]
//LB = 4: Size = 72, Time = 8.20602 msec, Performace = 18842.1 GFlop/s
//WB = 4: Size = 72, Time = 7.39876 msec, Performace = 20898   GFlop/s
//cuDNN-NCHW-implicit-prec: Size = 72, Time = 10.092 msec, Performance = 15320.9 GFlop/s
//cuDNN-NHWC-implicit-prec: Size = 72, Time =  9.502 msec, Performance = 16272.3 GFlop/s
//cuDNN-NHWC-GEMM-implicit: Size = 72, Time = 47.29 msec, Performance = 3269.6 GFlop/s

//[3] for: Feature = (48, 48), [N, IC, OC] = [128, 64, 64]
//LB = 4: Size = 40.5, Time = 4.39769 msec, Performace = 19777   GFlop/s
//WB = 4: Size = 40.5, Time = 3.98384 msec, Performace = 21831.5 GFlop/s
//cuDNN-NCHW-implicit-prec: Size = 40.5, Time = 5.758 msec, Performance = 15104.7 GFlop/s
//cuDNN-NHWC-implicit-prec: Size = 40.5, Time = 5.395 msec, Performance = 16121.1 GFlop/s
//cuDNN-NHWC-GEMM-implicit: Size = 40.5, Time = 26.525 msec, Performance = 3278.9 GFlop/s

//[4] for: Feature = (32, 32), [N, IC, OC] = [128, 128, 128]
//LB = 4: Size = 72, Time = 8.26207 msec, Performace = 18714.3 GFlop/s
//WB = 4: Size = 72, Time = 7.53276 msec, Performace = 20526.2 GFlop/s
//cuDNN-NCHW-implicit-prec: Size = 72, Time = 9.379 msec, Performance = 16485.6 GFlop/s
//cuDNN-NHWC-implicit-prec: Size = 72, Time = 9.505 msec, Performance = 16267.1 GFlop/s
//cuDNN-NHWC-GEMM-implicit: Size = 72, Time = 24.62 msec, Performance = 6280.2 GFlop/s

//[5] for: Feature = (24, 24), [N, IC, OC] = [128, 128, 128]
//LB = 4: Size = 40.5, Time = 4.37636 msec, Performace = 19873.4 GFlop/s
//WB = 4: Size = 40.5, Time = 4.00196 msec, Performace = 21732.6 GFlop/s
//cuDNN-NCHW-implicit-prec: Size = 40.5, Time = 5.423 msec, Performance = 16037.8 GFlop/s
//cuDNN-NHWC-implicit-prec: Size = 40.5, Time = 5.253 msec, Performance = 16556.8 GFlop/s
//cuDNN-NHWC-GEMM-implicit: Size = 40.5, Time = 13.85 msec, Performance = 6279.6 GFlop/s

//[6] for: Feature = (16, 16), [N, IC, OC] = [128, 256, 256]
//LB = 4: Size = 72, Time = 8.97014 msec, Performace = 17237.1 GFlop/s
//WB = 4: Size = 72, Time = 8.18056 msec, Performace = 18900.8 GFlop/s
//cuDNN-NCHW-implicit-prec: Size = 72, Time = 9.003 msec, Performance = 17174.1 GFlop/s
//cuDNN-NHWC-implicit-prec: Size = 72, Time = 9.149 msec, Performance = 16900.1 GFlop/s
//cuDNN-NHWC-GEMM-implicit: Size = 72, Time = 24.935 msec, Performance = 6200.9 GFlop/s

//[7] for: Feature = (12, 12), [N, IC, OC] = [128, 256, 256]
//LB = 4: Size = 40.5, Time = 4.4805  msec, Performace = 19411.4 GFlop/s
//WB = 4: Size = 40.5, Time = 4.01052 msec, Performace = 21686.2 GFlop/s
//cuDNN-NCHW-implicit-prec: Size = 40.5, Time = 5.352 msec, Performance = 16250.6 GFlop/s
//cuDNN-NHWC-implicit-prec: Size = 40.5, Time = 5.327 msec, Performance = 16326.8 GFlop/s
//cuDNN-NHWC-GEMM-implicit: Size = 40.5, Time = 13.76 msec, Performance = 6320.7 GFlop/s

//[8] for: Feature = (8, 8), [N, IC, OC] = [128, 512, 512]
//LB = 4: Size = 72, Time = 9.25448 msec, Performace = 16707.4 GFlop/s
//WB = 4: Size = 72, Time = 8.21645 msec, Performace = 18818.2 GFlop/s
//cuDNN-NCHW-implicit-prec: Size = 72, Time = 9.231 msec, Performance = 16750.0 GFlop/s
//cuDNN-NHWC-implicit-prec: Size = 72, Time = 9.174 msec, Performance = 16854.1 GFlop/s
//cuDNN-NHWC-GEMM-implicit: Size = 72, Time = 23.295 msec, Performance = 6637.4 GFlop/s

//[9] for: Feature = (6, 6), [N, IC, OC] = [128, 512, 512]
//LB = 4: Size = 40.5, Time = 4.73152 msec, Performace = 18381.6 GFlop/s
//WB = 4: Size = 40.5, Time = 4.11969 msec, Performace = 21111.6 GFlop/s
//cuDNN-NCHW-implicit-prec: Size = 40.5, Time = 5.439 msec, Performance = 15990.6 GFlop/s
//cuDNN-NHWC-implicit-prec: Size = 40.5, Time = 5.344 msec, Performance = 16274.9 GFlop/s
//cuDNN-NHWC-GEMM-implicit: Size = 40.5, Time = 13 msec, Performance = 6690.2 GFlop/s

#endif

//(1) FW = 6
//(2) sh = sw = 1
//(3) GN: (OC    ) % 64
//(4) GM: (N * OH) % 32
//(5) IC % 8 == 0
//(6) OW >= 3
#ifndef COMPILE
#define CONV_3D_WINOGRAD_S8_W6_64X32R_TEX
#endif
#ifndef CONV_3D_WINOGRAD_S8_W6_64X32R_TEX
#define CONV_3D_WINOGRAD_S8_W6_64X32R_TEX

template<int FH>
inline void conv3D_Winograd_s8_W6_64x32R_tex(jlong* streams, int &index, int length,
	const float*  X, cudaTextureObject_t texX, int IH, int IW,
	const float* CW,//[FH, FW, IC, OC]
	      float*  Y, int OH, int OW,
	int N, int IC, int OC,
	int ph, int pw,
	int GN, int GM, int &GNr, int &GMr)
{
	//======[Stage1 (OW % 3): Winograd_F(3, 6)]=========================================================================
	next_cudaStream(stream, streams, index, length);
	if ((pw <= 3) && (OW - IW - pw + 2 <= 0) && (OW >= 6)) {//pw <= 3
		conv3dWinograd_f3x6_ruse_k64x96R_p3_tex(stream, texX, IH, IW, CW, FH, Y, OH, OW, N, IC, OC, ph, pw);
		const int OWr = OW % 6;//align: OW % 3
		if (OWr >= 3) conv3dWinograd_f3x6_k64x96RC_tex(stream, (OW - OWr), texX, IH, IW, CW, FH, Y, OH, OW, N, IC, OC, ph, pw, OWr);
	}
#ifdef ENBALE_CONV3D_WINOGRAD_F3X6R_CHANNEL_TEMPLATE
	//OC = 2^x
	else if (OC ==  64) { if (IC ==  64) conv3dWinograd_f3x6_k64x96R_CT_tex(stream, texX, IH, IW, CW, FH, Y, OH, OW, N,  64,  64, ph, pw); else conv3dWinograd_f3x6_k64x96R_OCT_tex(stream, texX, IH, IW, CW, FH, Y, OH, OW, N, IC,  64, ph, pw); }
	else if (OC == 128) { if (IC == 128) conv3dWinograd_f3x6_k64x96R_CT_tex(stream, texX, IH, IW, CW, FH, Y, OH, OW, N, 128, 128, ph, pw); else conv3dWinograd_f3x6_k64x96R_OCT_tex(stream, texX, IH, IW, CW, FH, Y, OH, OW, N, IC, 128, ph, pw); }
	else if (OC == 256) { if (IC == 256) conv3dWinograd_f3x6_k64x96R_CT_tex(stream, texX, IH, IW, CW, FH, Y, OH, OW, N, 256, 256, ph, pw); else conv3dWinograd_f3x6_k64x96R_OCT_tex(stream, texX, IH, IW, CW, FH, Y, OH, OW, N, IC, 256, ph, pw); }
	else if (OC == 512) { if (IC == 512) conv3dWinograd_f3x6_k64x96R_CT_tex(stream, texX, IH, IW, CW, FH, Y, OH, OW, N, 512, 512, ph, pw); else conv3dWinograd_f3x6_k64x96R_OCT_tex(stream, texX, IH, IW, CW, FH, Y, OH, OW, N, IC, 512, ph, pw); }
	//OC = 64x
	else if (OC == 192) { if (IC == 192) conv3dWinograd_f3x6_k64x96R_CT_tex(stream, texX, IH, IW, CW, FH, Y, OH, OW, N, 192, 192, ph, pw); else conv3dWinograd_f3x6_k64x96R_OCT_tex(stream, texX, IH, IW, CW, FH, Y, OH, OW, N, IC, 192, ph, pw); }
	else if (OC == 320) { if (IC == 320) conv3dWinograd_f3x6_k64x96R_CT_tex(stream, texX, IH, IW, CW, FH, Y, OH, OW, N, 320, 320, ph, pw); else conv3dWinograd_f3x6_k64x96R_OCT_tex(stream, texX, IH, IW, CW, FH, Y, OH, OW, N, IC, 320, ph, pw); }
	else if (OC == 384) { if (IC == 384) conv3dWinograd_f3x6_k64x96R_CT_tex(stream, texX, IH, IW, CW, FH, Y, OH, OW, N, 384, 384, ph, pw); else conv3dWinograd_f3x6_k64x96R_OCT_tex(stream, texX, IH, IW, CW, FH, Y, OH, OW, N, IC, 384, ph, pw); }
	else if (OC == 448) { if (IC == 448) conv3dWinograd_f3x6_k64x96R_CT_tex(stream, texX, IH, IW, CW, FH, Y, OH, OW, N, 448, 448, ph, pw); else conv3dWinograd_f3x6_k64x96R_OCT_tex(stream, texX, IH, IW, CW, FH, Y, OH, OW, N, IC, 448, ph, pw); }
#endif
	else conv3dWinograd_f3x6_k64x96R_tex(stream, texX, IH, IW, CW, FH, Y, OH, OW, N, IC, OC, ph, pw);//common

	//======[Stage2 (OW % 2): Winograd_F(2, 3)]=========================================================================
	int OWr = OW % 3;
	if ((OWr >= 2) && !((N*OH) & 63)) {//Remainder: 2
		conv3dWinograd_SFW_f2x3_k64x128RC_tex(stream, (OW - OWr), texX, IH, IW, CW, FH, 6, Y, OH, OW, N, IC, OC, ph, pw, OWr);
		OWr = OWr & 1;//OWr % 2
	}

	//======[Stage3: GEMM]==============================================================================================
	if (OWr > 0) {//Remainder: 1
		const int GN64 = ((OC  ) >> 6 << 6);      //GN     = 64x
		const int GM32 = ((N*OH) >> 5 << 5) * OWr;//N  *OH = 32x
		index = 0;//save L2 cache: texX != X
		conv3D_Gemm_32x32RC(streams, index, length,
			X, IH, IW, CW, FH, 6, Y, OH, OW, OWr, IC, OC, 1, 1, ph, pw,
			GN64, GM32, 0, 0, (OW - OWr));
	}

	//=====[Process GNr and GMr]========================================================================================
	GNr = GN & 63;//64 channels
	GMr = ((N*OH) & 31)*OW;//32 groups
}

#endif

#endif


//FW = 7
#ifndef CONV_3D_WINOGRAD_S8_W7_64X32R_TEXTURE
#define CONV_3D_WINOGRAD_S8_W7_64X32R_TEXTURE

#ifndef CONV_3D_WINOGRAD_S8_W7_32X32R_P3_TEXTURE_BENCH_MARK_RTX3060TI
#define CONV_3D_WINOGRAD_S8_W7_32X32R_P3_TEXTURE_BENCH_MARK_RTX3060TI

//[0] for: Feature = (128, 128), [N, IC, OC] = [16, 64, 64]
//LB = 4: Size = 49, Time = 8.0124  msec, Performace = 13133 GFlop/s
//WB = 4: Size = 49, Time = 7.53933 msec, Performace = 13957 GFlop/s
//cuDNN-NCHW-GEMM-implicit-prec: Size = 49, Time = 6.594 msec, Performance = 15957.9 GFlop/s
//cuDNN-NHWC-GEMM-implicit-prec: Size = 49, Time = 6.404 msec, Performance = 16431.4 GFlop/s

//[1] for: Feature = (66, 66), [N, IC, OC] = [64, 64, 64]
//LB = 4: Size = 52.1104, Time = 8.55372 msec, Performace = 13082.7 GFlop/s
//WB = 4: Size = 52.1104, Time = 7.99853 msec, Performace = 13990.8 GFlop/s
//cuDNN-NCHW-GEMM-implicit-prec: Size = 52.110352, Time = 7.088 msec, Performance = 15788.1 GFlop/s
//cuDNN-NHWC-GEMM-implicit-prec: Size = 52.110352, Time = 6.766 msec, Performance = 16539.5 GFlop/s 

//[2] for: Feature = (64, 64), [N, IC, OC] = [64, 64, 64]
//LB = 4: Size = 49, Time = 8.26672 msec, Performace = 12728.9 GFlop/s
//WB = 4: Size = 49, Time = 7.62617 msec, Performace = 13798.1 GFlop/s 
//cuDNN-NCHW-GEMM-implicit-prec: Size = 49, Time = 6.622 msec, Performance = 15890.5 GFlop/s
//cuDNN-NHWC-GEMM-implicit-prec: Size = 49, Time = 6.406 msec, Performance = 16426.3 GFlop/s

//[3] for: Feature = (40, 40), [N, IC, OC] = [64, 128, 128]
//LB = 4: Size = 76.5625, Time = 12.9557 msec, Performace = 12690.7 GFlop/s
//WB = 4: Size = 76.5625, Time = 12.376  msec, Performace = 13285.1 GFlop/s
//cuDNN-NCHW-GEMM-implicit-prec: Size = 76.5625, Time = 10.068 msec, Performance = 16330.6 GFlop/s
//cuDNN-NHWC-GEMM-implicit-prec: Size = 76.5625, Time =  9.806 msec, Performance = 16766.9 GFlop/s

//[4] for: Feature = (34, 34), [N, IC, OC] = [64, 128, 128]
//LB = 4: Size = 55.3164, Time = 8.77886 msec, Performace = 13531.5 GFlop/s
//WB = 4: Size = 55.3164, Time = 8.71612 msec, Performace = 13628.9 GFlop/s
//cuDNN-NCHW-GEMM-implicit-prec: Size = 55.316406, Time = 7.375 msec, Performance = 16107.3 GFlop/s
//cuDNN-NHWC-GEMM-implicit-prec: Size = 55.316406, Time = 7.049 msec, Performance = 16852.2 GFlop/s

//[5] for: Feature = (32, 32), [N, IC, OC] = [64, 128, 128]
//LB = 4: Size = 49, Time = 8.01235 msec, Performace = 13133.1 GFlop/s
//WB = 4: Size = 49, Time = 7.51962 msec, Performace = 13993.6 GFlop/s
//cuDNN-NCHW-GEMM-implicit-prec: Size = 49, Time = 6.452 msec, Performance = 16309.2 GFlop/s
//cuDNN-NHWC-GEMM-implicit-prec: Size = 49, Time = 6.168 msec, Performance = 17060.1 GFlop/s

//[6] for: Feature = (18, 18), [N, IC, OC] = [64, 256, 256]
//LB = 4: Size = 62.015625, Time = 10.1938 msec, Performace = 13064.6 GFlop/s
//WB = 4: Size = 62.015625, Time = 9.47326 msec, Performace = 14058.3 GFlop/s
//cuDNN-NCHW-GEMM-implicit-prec: Size = 62.015625, Time = 8.278 msec, Performance = 16088.1 GFlop/s
//cuDNN-NHWC-GEMM-implicit-prec: Size = 62.015625, Time = 7.959 msec, Performance = 16732.9 GFlop/s

//[7] for: Feature = (16, 16), [N, IC, OC] = [64, 256, 256]
//LB = 4: Size = 49, Time = 7.86218 msec, Performace = 13383.9 GFlop/s
//WB = 4: Size = 49, Time = 7.34357 msec, Performace = 14329.1 GFlop/s
//cuDNN-NCHW-GEMM-implicit-prec: Size = 49, Time = 6.423 msec, Performance = 16382.8 GFlop/s
//cuDNN-NHWC-GEMM-implicit-prec: Size = 49, Time = 6.248 msec, Performance = 16841.7 GFlop/s

//[8] for: feature = (10, 10), [N, IC, OC] = [64, 512, 512]
//LB = 4: Size = 76.5625, Time = 13.2126 msec, Performace = 12444   GFlop/s
//WB = 4: Size = 76.5625, Time = 12.0092 msec, Performace = 13690.8 GFlop/s
//cuDNN-NCHW-GEMM-implicit-prec: Size = 76.5625, Time =  9.983 msec, Performance = 16469.7 GFlop/s
//cuDNN-NHWC-GEMM-implicit-prec: Size = 76.5625, Time = 10.545 msec, Performance = 15591.9 GFlop/s

//[9] for: feature = ( 8,  8), [N, IC, OC] = [64, 512, 512]
//LB = 4: Size = 49, Time = 8.36301 msec, Performace = 12582.4 GFlop/s
//WB = 4: Size = 49, Time = 7.46319 msec, Performace = 14099.4 GFlop/s
//cuDNN-NCHW-GEMM-implicit-prec: Size = 49, Time = 6.507 msec, Performance = 16171.3 GFlop/s
//cuDNN-NHWC-GEMM-implicit-prec: Size = 49, Time = 7.235 msec, Performance = 14544.1 GFlop/s

#endif

#ifndef CONV_3D_WINOGRAD_S8_W7_RUSE_32X32R_P3_TEXTURE_BENCH_MARK_RTX3060TI
#define CONV_3D_WINOGRAD_S8_W7_RUSE_32X32R_P3_TEXTURE_BENCH_MARK_RTX3060TI

//[0] for: Feature = (128, 128), [N, IC, OC] = [16, 64, 64]
//LB = 4: Size = 49, Time = 6.56454 msec, Performace = 16029.6 GFlop/s
//WB = 4: Size = 49, Time = 5.96151 msec, Performace = 17651   GFlop/s
//cuDNN-NCHW-GEMM-implicit-prec: Size = 49, Time = 6.594 msec, Performance = 15957.9 GFlop/s
//cuDNN-NHWC-GEMM-implicit-prec: Size = 49, Time = 6.404 msec, Performance = 16431.4 GFlop/s

//[1] for: Feature = (66, 66), [N, IC, OC] = [64, 64, 64]
//LB = 4: Size = 52.1104352, Time = 7.11009 msec, Performace = 15739.1 GFlop/s
//WB = 4: Size = 52.1104352, Time = 6.46012 msec, Performace = 17322.6 GFlop/s
//cuDNN-NCHW-GEMM-implicit-prec: Size = 52.110352, Time = 7.088 msec, Performance = 15788.1 GFlop/s
//cuDNN-NHWC-GEMM-implicit-prec: Size = 52.110352, Time = 6.766 msec, Performance = 16539.5 GFlop/s 

//[2] for: Feature = (64, 64), [N, IC, OC] = [64, 64, 64]
//LB = 4: Size = 49, Time = 6.63448 msec, Performace = 15860.6 GFlop/s
//WB = 4: Size = 49, Time = 5.95464 msec, Performace = 17671.4 GFlop/s
//cuDNN-NCHW-GEMM-implicit-prec: Size = 49, Time = 6.622 msec, Performance = 15890.5 GFlop/s
//cuDNN-NHWC-GEMM-implicit-prec: Size = 49, Time = 6.406 msec, Performance = 16426.3 GFlop/s

//[3] for: Feature = (40, 40), [N, IC, OC] = [64, 128, 128]
//LB = 4: Size = 76.5625, Time = 10.1792 msec, Performace = 16152.2 GFlop/s
//WB = 4: Size = 76.5625, Time = 9.24579 msec, Performace = 17782.9 GFlop/s
//cuDNN-NCHW-GEMM-implicit-prec: Size = 76.5625, Time = 10.068 msec, Performance = 16330.6 GFlop/s
//cuDNN-NHWC-GEMM-implicit-prec: Size = 76.5625, Time =  9.806 msec, Performance = 16766.9 GFlop/s

//[4] for: Feature = (34, 34), [N, IC, OC] = [64, 128, 128]
//LB = 4: Size = 55.316406, Time = 7.54602 msec, Performace = 15742.2 GFlop/s
//WB = 4: Size = 55.316406, Time = 6.88339 msec, Performace = 17257.6 GFlop/s
//cuDNN-NCHW-GEMM-implicit-prec: Size = 55.316406, Time = 7.375 msec, Performance = 16107.3 GFlop/s
//cuDNN-NHWC-GEMM-implicit-prec: Size = 55.316406, Time = 7.049 msec, Performance = 16852.2 GFlop/s

//[5] for: Feature = (32, 32), [N, IC, OC] = [64, 128, 128]
//LB = 4: Size = 49, Time = 6.45429 msec, Performace = 16303.4 GFlop/s
//WB = 4: Size = 49, Time = 5.9019  msec, Performace = 17829.3 GFlop/s
//cuDNN-NCHW-GEMM-implicit-prec: Size = 49, Time = 6.452 msec, Performance = 16309.2 GFlop/s
//cuDNN-NHWC-GEMM-implicit-prec: Size = 49, Time = 6.168 msec, Performance = 17060.1 GFlop/s

//[6] for: Feature = (18, 18), [N, IC, OC] = [64, 256, 256]
//LB = 4: Size = 62.015625, Time = 8.97078 msec, Performace = 14845.7 GFlop/s
//WB = 4: Size = 62.015625, Time = 8.17834 msec, Performace = 16284.2 GFlop/s
//cuDNN-NCHW-GEMM-implicit-prec: Size = 62.015625, Time = 8.278 msec, Performance = 16088.1 GFlop/s
//cuDNN-NHWC-GEMM-implicit-prec: Size = 62.015625, Time = 7.959 msec, Performance = 16732.9 GFlop/s

//[7] for: Feature = (16, 16), [N, IC, OC] = [64, 256, 256]
//LB = 4: Size = 49, Time = 6.78185 msec, Performace = 15515.9 GFlop/s
//WB = 4: Size = 49, Time = 6.19414 msec, Performace = 16988.1 GFlop/s
//cuDNN-NCHW-GEMM-implicit-prec: Size = 49, Time = 6.423 msec, Performance = 16382.8 GFlop/s
//cuDNN-NHWC-GEMM-implicit-prec: Size = 49, Time = 6.248 msec, Performance = 16841.7 GFlop/s

//[8] for: feature = (10, 10), [N, IC, OC] = [64, 512, 512]
//LB = 4: Size = 76.5625, Time = 11.7182 msec, Performace = 14030.8 GFlop/s
//WB = 4: Size = 76.5625, Time = 10.2219 msec, Performace = 16084.8 GFlop/s
//cuDNN-NCHW-GEMM-implicit-prec: Size = 76.5625, Time =  9.983 msec, Performance = 16469.7 GFlop/s
//cuDNN-NHWC-GEMM-implicit-prec: Size = 76.5625, Time = 10.545 msec, Performance = 15591.9 GFlop/s

//[9] for: feature = ( 8,  8), [N, IC, OC] = [64, 512, 512]
//LB = 4: Size = 76.5625, Time = 11.5998 msec, Performace = 14174.1 GFlop/s
//WB = 4: Size = 76.5625, Time = 10.1869 msec, Performace = 16140.1 GFlop/s
//cuDNN-NCHW-GEMM-implicit-prec: Size = 49, Time = 6.507 msec, Performance = 16171.3 GFlop/s
//cuDNN-NHWC-GEMM-implicit-prec: Size = 49, Time = 7.235 msec, Performance = 14544.1 GFlop/s

#endif

//(1) FW = 7
//(2) sh = sw = 1
//(3) GN: (OC    ) % 64
//(4) GM: (N * OH) % 32
//(5) IC % 8 == 0
//(6) OW >= 2
#ifndef COMPILE
#define CONV_3D_WINOGRAD_S8_W7_64X32R_TEX
#endif
#ifndef CONV_3D_WINOGRAD_S8_W7_64X32R_TEX
#define CONV_3D_WINOGRAD_S8_W7_64X32R_TEX

template<int FH>
inline void conv3D_Winograd_s8_W7_64x32R_tex(jlong* streams, int &index, int length,
	const float*  X, cudaTextureObject_t texX, int IH, int IW,
	const float* CW,//[FH, FW, IC, OC]
	      float*  Y, int OH, int OW,
	int N, int IC, int OC,
	int ph, int pw,
	int GN, int GM, int &GNr, int &GMr)
{
	//======[Stage1 (OW % 2): Winograd_F(2, 7)]=========================================================================
	next_cudaStream(stream, streams, index, length);
	if ((pw <= 3) && (OW - IW - pw + 3 <= 0) && (OW >= 4)) {//pw <= 3
#ifdef ENBALE_CONV3D_WINOGRAD_F2X7R_CHANNEL_TEMPLATE
		//IC = 2^x
		if      (IC ==  64 && OC ==  64) conv3dWinograd_f2x7_ruse_k64x64R_p3_CT_tex(stream, texX, IH, IW, CW, FH, Y, OH, OW, N,  64,  64, ph, pw);
		else if (IC == 128 && OC == 128) conv3dWinograd_f2x7_ruse_k64x64R_p3_CT_tex(stream, texX, IH, IW, CW, FH, Y, OH, OW, N, 128, 128, ph, pw);
		else if (IC == 256 && OC == 256) conv3dWinograd_f2x7_ruse_k64x64R_p3_CT_tex(stream, texX, IH, IW, CW, FH, Y, OH, OW, N, 256, 256, ph, pw);
		else if (IC == 512 && OC == 512) conv3dWinograd_f2x7_ruse_k64x64R_p3_CT_tex(stream, texX, IH, IW, CW, FH, Y, OH, OW, N, 512, 512, ph, pw);
		//IC = 64x
		else if (IC == 192 && OC == 192) conv3dWinograd_f2x7_ruse_k64x64R_p3_CT_tex(stream, texX, IH, IW, CW, FH, Y, OH, OW, N, 192, 192, ph, pw);
		else if (IC == 320 && OC == 320) conv3dWinograd_f2x7_ruse_k64x64R_p3_CT_tex(stream, texX, IH, IW, CW, FH, Y, OH, OW, N, 320, 320, ph, pw);
		else if (IC == 384 && OC == 384) conv3dWinograd_f2x7_ruse_k64x64R_p3_CT_tex(stream, texX, IH, IW, CW, FH, Y, OH, OW, N, 384, 384, ph, pw);
		else if (IC == 448 && OC == 448) conv3dWinograd_f2x7_ruse_k64x64R_p3_CT_tex(stream, texX, IH, IW, CW, FH, Y, OH, OW, N, 448, 448, ph, pw);
		else
#endif
		conv3dWinograd_f2x7_ruse_k64x64R_p3_tex(stream, texX, IH, IW, CW, FH, Y, OH, OW, N, IC, OC, ph, pw);//common

		const int OWr = OW & 3;//align: OW % 2
		if (OWr >= 2) { const int ow_index = OW - OWr;
#ifdef ENBALE_CONV3D_WINOGRAD_F2X7R_CHANNEL_TEMPLATE
			//IC = 2^x
			if      (IC ==  64 && OC ==  64) conv3dWinograd_f2x7_k64x64RC_CT_tex(stream, ow_index, texX, IH, IW, CW, FH, Y, OH, OW, N,  64,  64, ph, pw, OWr);
			else if (IC == 128 && OC == 128) conv3dWinograd_f2x7_k64x64RC_CT_tex(stream, ow_index, texX, IH, IW, CW, FH, Y, OH, OW, N, 128, 128, ph, pw, OWr);
			else if (IC == 256 && OC == 256) conv3dWinograd_f2x7_k64x64RC_CT_tex(stream, ow_index, texX, IH, IW, CW, FH, Y, OH, OW, N, 256, 256, ph, pw, OWr);
			else if (IC == 512 && OC == 512) conv3dWinograd_f2x7_k64x64RC_CT_tex(stream, ow_index, texX, IH, IW, CW, FH, Y, OH, OW, N, 512, 512, ph, pw, OWr);
			//IC = 64x
			else if (IC == 192 && OC == 192) conv3dWinograd_f2x7_k64x64RC_CT_tex(stream, ow_index, texX, IH, IW, CW, FH, Y, OH, OW, N, 192, 192, ph, pw, OWr);
			else if (IC == 320 && OC == 320) conv3dWinograd_f2x7_k64x64RC_CT_tex(stream, ow_index, texX, IH, IW, CW, FH, Y, OH, OW, N, 320, 320, ph, pw, OWr);
			else if (IC == 384 && OC == 384) conv3dWinograd_f2x7_k64x64RC_CT_tex(stream, ow_index, texX, IH, IW, CW, FH, Y, OH, OW, N, 384, 384, ph, pw, OWr);
			else if (IC == 448 && OC == 448) conv3dWinograd_f2x7_k64x64RC_CT_tex(stream, ow_index, texX, IH, IW, CW, FH, Y, OH, OW, N, 448, 448, ph, pw, OWr); 
			else
#endif
			conv3dWinograd_f2x7_k64x64RC_tex(stream, ow_index, texX, IH, IW, CW, FH, Y, OH, OW, N, IC, OC, ph, pw, OWr);//common
		}
	}
#ifdef ENBALE_CONV3D_WINOGRAD_F2X7R_CHANNEL_TEMPLATE
	//IC = 2^x
	else if (IC ==  64 && OC ==  64) conv3dWinograd_f2x7_k64x64R_CT_tex(stream, texX, IH, IW, CW, FH, Y, OH, OW, N,  64,  64, ph, pw);
	else if (IC == 128 && OC == 128) conv3dWinograd_f2x7_k64x64R_CT_tex(stream, texX, IH, IW, CW, FH, Y, OH, OW, N, 128, 128, ph, pw);
	else if (IC == 256 && OC == 256) conv3dWinograd_f2x7_k64x64R_CT_tex(stream, texX, IH, IW, CW, FH, Y, OH, OW, N, 256, 256, ph, pw);
	else if (IC == 512 && OC == 512) conv3dWinograd_f2x7_k64x64R_CT_tex(stream, texX, IH, IW, CW, FH, Y, OH, OW, N, 512, 512, ph, pw);
	//IC = 64x
	else if (IC == 192 && OC == 192) conv3dWinograd_f2x7_k64x64R_CT_tex(stream, texX, IH, IW, CW, FH, Y, OH, OW, N, 192, 192, ph, pw);
	else if (IC == 320 && OC == 320) conv3dWinograd_f2x7_k64x64R_CT_tex(stream, texX, IH, IW, CW, FH, Y, OH, OW, N, 320, 320, ph, pw);
	else if (IC == 384 && OC == 384) conv3dWinograd_f2x7_k64x64R_CT_tex(stream, texX, IH, IW, CW, FH, Y, OH, OW, N, 384, 384, ph, pw);
	else if (IC == 448 && OC == 448) conv3dWinograd_f2x7_k64x64R_CT_tex(stream, texX, IH, IW, CW, FH, Y, OH, OW, N, 448, 448, ph, pw);
#endif
	else conv3dWinograd_f2x7_k64x64R_tex(stream, texX, IH, IW, CW, FH, Y, OH, OW, N, IC, OC, ph, pw);//common

	//======[Stage2: GEMM]==============================================================================================
	const int OWr = OW & 1;
	if (OWr > 0) {//Remainder: 1
		const int GN64 = ((OC  ) >> 6 << 6);      //GN     = 64x
		const int GM32 = ((N*OH) >> 5 << 5) * OWr;//N * OH = 32x
		index = 0;//save L2 cache: texX ! = X
		conv3D_Gemm_32x32RC(streams, index, length, 
			X, IH, IW, CW, FH, 7, Y, OH, OW, OWr, IC, OC, 1, 1, ph, pw,
			GN64, GM32, 0, 0, OW - OWr);
	}

	//=====[Process GNr and GMr]========================================================================================
	GNr = GN & 63;//64 channels
	GMr = ((N*OH) & 31)*OW;//32 groups
}

#endif

#endif

#endif



//------[state = 16]-----------------------------
#ifndef CONV_3D_WINOGRAD_S16_AREA
#define CONV_3D_WINOGRAD_S16_AREA

//FW = 7
#ifndef CONV_3D_WINOGRAD_S16_W7_32X32R_P3_TEXTURE
#define CONV_3D_WINOGRAD_S16_W7_32X32R_P3_TEXTURE

#ifndef CONV_3D_WINOGRAD_S16_W7_32X32R_P3_TEXTURE_BENCH_MARK_RTX3060TI
#define CONV_3D_WINOGRAD_S16_W7_32X32R_P3_TEXTURE_BENCH_MARK_RTX3060TI

//[0] for: Feature = (128, 128), [N, IC, OC] = [32, 64, 64]
//LB = 4: Size = 98, Time = 8.66155 msec, Performace = 24297.4 GFlop/s
//WB = 4: Size = 98, Time = 8.12108 msec, Performace = 25914.5 GFlop/s
//cuDNN-NCHW-GEMM-implicit-prec: Size = 98, Time = 12.691 msec, Performance = 16582.9 GFlop/s
//cuDNN-NHWC-GEMM-implicit-prec: Size = 98, Time = 12.818 msec, Performance = 16418.6 GFlop/s
//cuDNN-NHWC-GEMM-implicit: Size = 98, Time = 64.91 msec, Performance = 3242.2 GFlop/s

//[1] for: Feature = (120, 120), [N, IC, OC] = [32, 64, 64]
//LB = 4: Size = 86.132813, Time = 7.34329 msec, Performace = 25188.8 GFlop/s
//WB = 4: Size = 86.132813, Time = 6.92344 msec, Performace = 26716.3 GFlop/s
//cuDNN-NCHW-GEMM-implicit-prec: Size = 86.132813, Time = 11.801 msec, Performance = 15674.0 GFlop/s
//cuDNN-NHWC-GEMM-implicit-prec: Size = 86.132813, Time = 11.084 msec, Performance = 16687.9 GFlop/s
//cuDNN-NHWC-GEMM-implicit: Size = 86.132813, Time = 56.57 msec, Performance = 3269.7 GFlop/s

//[2] for: Feature = (112, 112), [N, IC, OC] = [64, 64, 64]
//LB = 4: Size = 150.0625, Time = 13.1449 msec, Performace = 24515.7 GFlop/s
//WB = 4: Size = 150.0625, Time = 12.3088 msec, Performace = 26181   GFlop/s
//cuDNN-NCHW-GEMM-implicit-prec: Size = 150.0625, Time = 19.498 msec, Performance = 16527.7 GFlop/s
//cuDNN-NHWC-GEMM-implicit-prec: Size = 150.0625, Time = 19.264 msec, Performance = 16728.4 GFlop/s
//cuDNN-NHWC-GEMM-implicit: Size = 150.0625, Time = 100.12 msec, Performance = 3218.7 GFlop/s

//[3] for: Feature = (80, 80), [N, IC, OC] = [64, 64, 64]
//LB = 4: Size = 76.5625, Time = 6.48101 msec, Performace = 25369   GFlop/s
//WB = 4: Size = 76.5625, Time = 6.08989 msec, Performace = 26998.3 GFlop/s
//cuDNN-NCHW-GEMM-implicit-prec: Size = 76.5625, Time = 10.575 msec, Performance = 15547.7 GFlop/s
//cuDNN-NHWC-GEMM-implicit-prec: Size = 76.5625, Time =  9.894 msec, Performance = 16617.8 GFlop/s
//cuDNN-NHWC-GEMM-implicit: Size = 153.125, Time = 100.945 msec, Performance = 3257.6 GFlop/s

//[4] for: Feature = (64, 64), [N, IC, OC] = [128, 64, 64]
//LB = 4: Size = 98, Time = 8.60688 msec, Performace = 24451.8 GFlop/s
//WB = 4: Size = 98, Time = 8.07407 msec, Performace = 26065.3 GFlop/s
//cuDNN-NCHW-GEMM-implicit-prec: Size = 98, Time = 12.799 msec, Performance = 16443.0 GFlop/s
//cuDNN-NHWC-GEMM-implicit-prec: Size = 98, Time = 12.594 msec, Performance = 16710.6 GFlop/s
//cuDNN-NHWC-GEMM-implicit: Size = 98, Time = 64.58 msec, Performance = 3258.8 GFlop/s

//[5] for: Feature = (40, 40), [N, IC, OC] = [64, 128, 128]
//LB = 4: Size = 76.5625, Time = 7.66704 msec, Performace = 21444.6 GFlop/s
//WB = 4: Size = 76.5625, Time = 7.22939 msec, Performace = 22742.8 GFlop/s
//cuDNN-NCHW-GEMM-implicit-prec: Size = 76.5625, Time = 10.225 msec, Performance = 16079.9 GFlop/s
//cuDNN-NHWC-GEMM-implicit-prec: Size = 76.5625, Time =  9.861 msec, Performance = 16673.4 GFlop/s
//cuDNN-NHWC-GEMM-implicit: Size = 153.125, Time = 53.279 msec, Performance = 6171.8 GFlop/s

//[6] for: Feature = (32, 32), [N, IC, OC] = [128, 128, 128]
//LB = 4: Size = 98, Time = 10.4474 msec, Performace = 20144.2 GFlop/s
//WB = 4: Size = 98, Time =  9.8671 msec, Performace = 21328.8 GFlop/s
//cuDNN-NCHW-GEMM-implicit-prec: Size = 98, Time = 12.242 msec, Performance = 17191.1 GFlop/s
//cuDNN-NHWC-GEMM-implicit-prec: Size = 98, Time = 12.264 msec, Performance = 17160.3 GFlop/s
//cuDNN-NHWC-GEMM-implicit: Size = 98, Time = 36.375 msec, Performance = 5785.7 GFlop/s

//[7] for: Feature = (20, 20), [N, IC, OC] = [64, 256, 256]
//LB = 4: Size = 76.5625, Time = 7.52199 msec, Performace = 21858.1 GFlop/s
//WB = 4: Size = 76.5625, Time = 7.1563  msec, Performace = 22975.1 GFlop/s
//cuDNN-NCHW-GEMM-implicit-prec: Size = 76.5625, Time = 10.194 msec, Performance = 16128.8 GFlop/s
//cuDNN-NHWC-GEMM-implicit-prec: Size = 76.5625, Time =  9.734 msec, Performance = 16891.0 GFlop/s
//cuDNN-NHWC-GEMM-implicit: Size = 153.125, Time = 57.285 msec, Performance = 5740.3 GFlop/s

//[8] for: Feature = (16, 16), [N, IC, OC] = [128, 256, 256]
//LB = 4: Size = 98, Time = 12.2646 msec, Performace = 17159.4 GFlop/s
//WB = 4: Size = 98, Time = 11.4454 msec, Performace = 18387.6 GFlop/s
//cuDNN-NCHW-GEMM-implicit-prec: Size = 98, Time = 12.409 msec, Performance = 16959.7 GFlop/s
//cuDNN-NHWC-GEMM-implicit-prec: Size = 98, Time = 12.407 msec, Performance = 16962.5 GFlop/s
//cuDNN-NHWC-GEMM-implicit: Size = 98, Time = 35.965 msec, Performance = 5851.6 GFlop/s

//[9] for: Feature = (10, 10), [N, IC, OC] = [64, 512, 512]
//LB = 4: Size = 76.5625, Time = 7.51321 msec, Performace = 21883.7 GFlop/s
//WB = 4: Size = 76.5625, Time = 6.94956 msec, Performace = 23658.6 GFlop/s
//cuDNN-NCHW-GEMM-implicit-prec: Size = 76.5625, Time = 10.141 msec, Performance = 16213.1 GFlop/s
//cuDNN-NHWC-GEMM-implicit-prec: Size = 76.5625, Time = 10.751 msec, Performance = 15293.2 GFlop/s
//cuDNN-NHWC-GEMM-implicit: Size = 86.132813, Time = 57.185001 msec, Performance = 3234.6 GFlop/s

#endif

//(1) FW = 7
//(2) sh = sw = 1
//(3) GN: (OC    ) % 32
//(4) GM: (N * OH) % 32
//(5) IC % 8 == 0
//(6) pw <= 3 && IW >= OW
//(7) OW >= 10
#ifndef COMPILE
#define CONV_3D_WINOGRAD_S16_W7_32X32R_P3_TEX
#endif
#ifndef CONV_3D_WINOGRAD_S16_W7_32X32R_P3_TEX
#define CONV_3D_WINOGRAD_S16_W7_32X32R_P3_TEX

template<int FH>
inline void conv3D_Winograd_s16_W7_32x32R_p3_tex(jlong* streams, int &index, int length,
	const float*  X, cudaTextureObject_t texX, int IH, int IW,
	const float* CW,//[FH, FW, IC, OC]
	      float*  Y, int OH, int OW,
	int N, int IC, int OC,
	int ph, int pw,
	int GN, int GM, int &GNr, int &GMr)
{
	//======[Stage1 (OW % 10): Winograd_F(10, 7) ]===========================================================
	next_cudaStream(stream, streams, index, length);
#ifdef ENBALE_CONV3D_WINOGRAD_FAX7R_CHANNEL_TEMPLATE
	//IC = 2^x
	if      (IC ==  32 && OC ==  32) conv3dWinograd_f10x7_k32x320R_p3_CT_tex(stream, texX, IH, IW, CW, FH, Y, OH, OW, N,  32,  32, ph, pw);
	else if (IC ==  64 && OC ==  64) conv3dWinograd_f10x7_k32x320R_p3_CT_tex(stream, texX, IH, IW, CW, FH, Y, OH, OW, N,  64,  64, ph, pw);
	else if (IC == 128 && OC == 128) conv3dWinograd_f10x7_k32x320R_p3_CT_tex(stream, texX, IH, IW, CW, FH, Y, OH, OW, N, 128, 128, ph, pw);
	else if (IC == 256 && OC == 256) conv3dWinograd_f10x7_k32x320R_p3_CT_tex(stream, texX, IH, IW, CW, FH, Y, OH, OW, N, 256, 256, ph, pw);
	else if (IC == 512 && OC == 512) conv3dWinograd_f10x7_k32x320R_p3_CT_tex(stream, texX, IH, IW, CW, FH, Y, OH, OW, N, 512, 512, ph, pw);
	//IC = 32x
	else if (IC == 192 && OC == 192) conv3dWinograd_f10x7_k32x320R_p3_CT_tex(stream, texX, IH, IW, CW, FH, Y, OH, OW, N, 192, 192, ph, pw);
	else if (IC == 224 && OC == 224) conv3dWinograd_f10x7_k32x320R_p3_CT_tex(stream, texX, IH, IW, CW, FH, Y, OH, OW, N, 224, 224, ph, pw);
	else if (IC == 320 && OC == 320) conv3dWinograd_f10x7_k32x320R_p3_CT_tex(stream, texX, IH, IW, CW, FH, Y, OH, OW, N, 320, 320, ph, pw);
	else if (IC == 384 && OC == 384) conv3dWinograd_f10x7_k32x320R_p3_CT_tex(stream, texX, IH, IW, CW, FH, Y, OH, OW, N, 384, 384, ph, pw);
	else if (IC == 448 && OC == 448) conv3dWinograd_f10x7_k32x320R_p3_CT_tex(stream, texX, IH, IW, CW, FH, Y, OH, OW, N, 448, 448, ph, pw);
	else
#endif
	conv3dWinograd_f10x7_k32x320R_p3_tex(stream, texX, IH, IW, CW, FH, Y, OH, OW, N, IC, OC, ph, pw);//common

	//======[Stage2 (OW % 4): Winograd_F(2 + 2, 7) ]==========================================================
	int OWr = OW % 10;
	if ((OWr >= 4) && !(OC & 63)) {//Remainder: 4, 5, 6, 7, 8, 9
		const int ow_index = OW - OWr;
#ifdef ENBALE_CONV3D_WINOGRAD_FAX7R_CHANNEL_TEMPLATE
		//IC = 2^x
		if      (IC ==  64 && OC ==  64) conv3dWinograd_f2x7_ruse_k64x64RC_p3_CT_tex(stream, ow_index, texX, IH, IW, CW, FH, Y, OH, OW, N,  64,  64, ph, pw, OWr);
		else if (IC == 128 && OC == 128) conv3dWinograd_f2x7_ruse_k64x64RC_p3_CT_tex(stream, ow_index, texX, IH, IW, CW, FH, Y, OH, OW, N, 128, 128, ph, pw, OWr);
		else if (IC == 256 && OC == 256) conv3dWinograd_f2x7_ruse_k64x64RC_p3_CT_tex(stream, ow_index, texX, IH, IW, CW, FH, Y, OH, OW, N, 256, 256, ph, pw, OWr);
		else if (IC == 512 && OC == 512) conv3dWinograd_f2x7_ruse_k64x64RC_p3_CT_tex(stream, ow_index, texX, IH, IW, CW, FH, Y, OH, OW, N, 512, 512, ph, pw, OWr);
		//IC = 64x
		else if (IC == 192 && OC == 192) conv3dWinograd_f2x7_ruse_k64x64RC_p3_CT_tex(stream, ow_index, texX, IH, IW, CW, FH, Y, OH, OW, N, 192, 192, ph, pw, OWr);
		else if (IC == 320 && OC == 320) conv3dWinograd_f2x7_ruse_k64x64RC_p3_CT_tex(stream, ow_index, texX, IH, IW, CW, FH, Y, OH, OW, N, 320, 320, ph, pw, OWr);
		else if (IC == 384 && OC == 384) conv3dWinograd_f2x7_ruse_k64x64RC_p3_CT_tex(stream, ow_index, texX, IH, IW, CW, FH, Y, OH, OW, N, 384, 384, ph, pw, OWr);
		else if (IC == 448 && OC == 448) conv3dWinograd_f2x7_ruse_k64x64RC_p3_CT_tex(stream, ow_index, texX, IH, IW, CW, FH, Y, OH, OW, N, 448, 448, ph, pw, OWr);
		else
#endif
		conv3dWinograd_f2x7_ruse_k64x64RC_p3_tex(stream, ow_index, texX, IH, IW, CW, FH, Y, OH, OW, N, IC, OC, ph, pw, OWr);//common
		OWr = OWr & 3;//OWr % 4
	}

	//======[Stage3 (OW % 2): Winograd_F(2, 7) ]=============================================================
	if ((OWr >= 2) && !(OC & 63)) {//Remainder: 2, 3
		const int ow_index = OW - OWr;
#ifdef ENBALE_CONV3D_WINOGRAD_FAX7R_CHANNEL_TEMPLATE
		//IC = 2^x
		if      (IC ==  64 && OC ==  64) conv3dWinograd_f2x7_k64x64RC_CT_tex(stream, ow_index, texX, IH, IW, CW, FH, Y, OH, OW, N,  64,  64, ph, pw, OWr);
		else if (IC == 128 && OC == 128) conv3dWinograd_f2x7_k64x64RC_CT_tex(stream, ow_index, texX, IH, IW, CW, FH, Y, OH, OW, N, 128, 128, ph, pw, OWr);
		else if (IC == 256 && OC == 256) conv3dWinograd_f2x7_k64x64RC_CT_tex(stream, ow_index, texX, IH, IW, CW, FH, Y, OH, OW, N, 256, 256, ph, pw, OWr);
		else if (IC == 512 && OC == 512) conv3dWinograd_f2x7_k64x64RC_CT_tex(stream, ow_index, texX, IH, IW, CW, FH, Y, OH, OW, N, 512, 512, ph, pw, OWr);
		//IC = 64x
		else if (IC == 192 && OC == 192) conv3dWinograd_f2x7_k64x64RC_CT_tex(stream, ow_index, texX, IH, IW, CW, FH, Y, OH, OW, N, 192, 192, ph, pw, OWr);
		else if (IC == 320 && OC == 320) conv3dWinograd_f2x7_k64x64RC_CT_tex(stream, ow_index, texX, IH, IW, CW, FH, Y, OH, OW, N, 320, 320, ph, pw, OWr);
		else if (IC == 448 && OC == 448) conv3dWinograd_f2x7_k64x64RC_CT_tex(stream, ow_index, texX, IH, IW, CW, FH, Y, OH, OW, N, 448, 448, ph, pw, OWr);
		else 
#endif
		conv3dWinograd_f2x7_k64x64RC_tex(stream, ow_index, texX, IH, IW, CW, FH, Y, OH, OW, N, IC, OC, ph, pw, OWr);//common
		OWr = OWr & 1;//OWr % 2
	}

	//======[Stage4: GEMM]===================================================================================
	if (OWr > 0) {//Remainder: 1
		const int GN32 = ((OC  ) >> 5 << 5);      //GN   = 32x
		const int GM32 = ((N*OH) >> 5 << 5) * OWr;//N*OH = 32x
		index = 0;//save L2 cache: texX ! = X
		conv3D_Gemm_32x32RC(streams, index, length,
			X, IH, IW, CW, FH, 7, Y, OH, OW, OWr, IC, OC, 1, 1, ph, pw,
			GN32, GM32, 0, 0, OW - OWr);
	}

	//=====[Process GNr and GMr]=============================================================================
	GNr = GN & 31;         //32 channels
	GMr = ((N*OH) & 31)*OW;//32 groups
}

#endif

#endif


//FW = 7
#ifndef CONV_3D_WINOGRAD_S16_W7_64X32R_P3_TEXTURE
#define CONV_3D_WINOGRAD_S16_W7_64X32R_P3_TEXTURE

#ifndef CONV_3D_WINOGRAD_S16_W7_64X32R_P3_TEXTURE_BENCH_MARK_RTX3060TI
#define CONV_3D_WINOGRAD_S16_W7_64X32R_P3_TEXTURE_BENCH_MARK_RTX3060TI

//[0] for: Feature = (128, 128), [N, IC, OC] = [32, 64, 64]
//LB = 4: Size = 98, Time = 8.03653 msec, Performace = 26187.1 GFlop/s
//WB = 4: Size = 98, Time = 7.35944 msec, Performace = 28596.4 GFlop/s
//cuDNN-NCHW-GEMM-implicit-prec: Size = 98, Time = 12.691 msec, Performance = 16582.9 GFlop/s
//cuDNN-NHWC-GEMM-implicit-prec: Size = 98, Time = 12.818 msec, Performance = 16418.6 GFlop/s
//cuDNN-NHWC-GEMM-implicit: Size = 98, Time = 64.91 msec, Performance = 3242.2 GFlop/s

//[1] for: Feature = (120, 120), [N, IC, OC] = [32, 64, 64]
//LB = 4: Size = 86.132813, Time = 6.62705 msec, Performace = 27911.2 GFlop/s
//WB = 4: Size = 86.132813, Time = 6.10991 msec, Performace = 30273.6 GFlop/s
//cuDNN-NCHW-GEMM-implicit-prec: Size = 86.132813, Time = 11.801 msec, Performance = 15674.0 GFlop/s
//cuDNN-NHWC-GEMM-implicit-prec: Size = 86.132813, Time = 11.084 msec, Performance = 16687.9 GFlop/s
//cuDNN-NHWC-GEMM-implicit: Size = 86.132813, Time = 56.57 msec, Performance = 3269.7 GFlop/s

//[2] for: Feature = (112, 112), [N, IC, OC] = [64, 64, 64]
//LB = 4: Size = 150.0625, Time = 12.0051 msec, Performace = 26843.4 GFlop/s
//WB = 4: Size = 150.0625, Time = 11.0262 msec, Performace = 29226.5 GFlop/s
//cuDNN-NCHW-GEMM-implicit-prec: Size = 150.0625, Time = 19.498 msec, Performance = 16527.7 GFlop/s
//cuDNN-NHWC-GEMM-implicit-prec: Size = 150.0625, Time = 19.264 msec, Performance = 16728.4 GFlop/s
//cuDNN-NHWC-GEMM-implicit: Size = 150.0625, Time = 100.12 msec, Performance = 3218.7 GFlop/s

//[3] for: Feature = (80, 80), [N, IC, OC] = [64, 64, 64]
//LB = 4: Size = 76.5625, Time = 5.89455 msec, Performace = 27893   GFlop/s
//WB = 4: Size = 76.5625, Time = 5.4335  msec, Performace = 30259.8 GFlop/s
//cuDNN-NCHW-GEMM-implicit-prec: Size = 76.5625, Time = 10.575 msec, Performance = 15547.7 GFlop/s
//cuDNN-NHWC-GEMM-implicit-prec: Size = 76.5625, Time =  9.894 msec, Performance = 16617.8 GFlop/s
//cuDNN-NHWC-GEMM-implicit: Size = 153.125, Time = 100.945 msec, Performance = 3257.6 GFlop/s

//[4] for: Feature = (64, 64), [N, IC, OC] = [128, 64, 64]
//LB = 4: Size = 98, Time = 8.08346 msec, Performace = 26035.1 GFlop/s
//WB = 4: Size = 98, Time = 7.416   msec, Performace = 28378.3 GFlop/s
//cuDNN-NCHW-GEMM-implicit-prec: Size = 98, Time = 12.799 msec, Performance = 16443.0 GFlop/s
//cuDNN-NHWC-GEMM-implicit-prec: Size = 98, Time = 12.594 msec, Performance = 16710.6 GFlop/s
//cuDNN-NHWC-GEMM-implicit: Size = 98, Time = 64.58 msec, Performance = 3258.8 GFlop/s

//[5] for: Feature = (40, 40), [N, IC, OC] = [64, 128, 128]
//LB = 4: Size = 76.5625, Time = 6.41693 msec, Performace = 25622.3 GFlop/s
//WB = 4: Size = 76.5625, Time = 5.85207 msec, Performace = 28095.5 GFlop/s
//cuDNN-NCHW-GEMM-implicit-prec: Size = 76.5625, Time = 10.225 msec, Performance = 16079.9 GFlop/s
//cuDNN-NHWC-GEMM-implicit-prec: Size = 76.5625, Time =  9.861 msec, Performance = 16673.4 GFlop/s
//cuDNN-NHWC-GEMM-implicit: Size = 153.125, Time = 53.279 msec, Performance = 6171.8 GFlop/s

//[6] for: Feature = (32, 32), [N, IC, OC] = [128, 128, 128]
//LB = 4: Size = 98, Time = 8.74437 msec, Performace = 24067.3 GFlop/s
//WB = 4: Size = 98, Time = 8.06393 msec, Performace = 26098.1 GFlop/s
//cuDNN-NCHW-GEMM-implicit-prec: Size = 98, Time = 12.242 msec, Performance = 17191.1 GFlop/s
//cuDNN-NHWC-GEMM-implicit-prec: Size = 98, Time = 12.264 msec, Performance = 17160.3 GFlop/s
//cuDNN-NHWC-GEMM-implicit: Size = 98, Time = 36.375 msec, Performance = 5785.7 GFlop/s

//[7] for: Feature = (20, 20), [N, IC, OC] = [64, 256, 256]
//LB = 4: Size = 76.5625, Time = 6.12998 msec, Performace = 26821.7 GFlop/s
//WB = 4: Size = 76.5625, Time = 5.67085 msec, Performace = 28993.3 GFlop/s
//cuDNN-NCHW-GEMM-implicit-prec: Size = 76.5625, Time = 10.194 msec, Performance = 16128.8 GFlop/s
//cuDNN-NHWC-GEMM-implicit-prec: Size = 76.5625, Time =  9.734 msec, Performance = 16891.0 GFlop/s
//cuDNN-NHWC-GEMM-implicit: Size = 153.125, Time = 57.285 msec, Performance = 5740.3 GFlop/s

//[8] for: Feature = (16, 16), [N, IC, OC] = [128, 256, 256]
//LB = 4: Size = 98, Time = 10.8016 msec, Performace = 19483.5 GFlop/s
//WB = 4: Size = 98, Time = 9.92577 msec, Performace = 21202.7 GFlop/s
//cuDNN-NCHW-GEMM-implicit-prec: Size = 98, Time = 12.409 msec, Performance = 16959.7 GFlop/s
//cuDNN-NHWC-GEMM-implicit-prec: Size = 98, Time = 12.407 msec, Performance = 16962.5 GFlop/s
//cuDNN-NHWC-GEMM-implicit: Size = 98, Time = 35.965 msec, Performance = 5851.6 GFlop/s

//[9] for: Feature = (10, 10), [N, IC, OC] = [64, 512, 512]
//LB = 4: Size = 76.5625, Time = 6.37006 msec, Performace = 25810.9 GFlop/s
//WB = 4: Size = 76.5625, Time = 5.83609 msec, Performace = 28172.4 GFlop/s
//cuDNN-NCHW-GEMM-implicit-prec: Size = 76.5625, Time = 10.141 msec, Performance = 16213.1 GFlop/s
//cuDNN-NHWC-GEMM-implicit-prec: Size = 76.5625, Time = 10.751 msec, Performance = 15293.2 GFlop/s
//cuDNN-NHWC-GEMM-implicit: Size = 86.132813, Time = 57.185001 msec, Performance = 3234.6 GFlop/s

#endif

//(1) FW = 7
//(2) sh = sw = 1
//(3) GN: (OC    ) % 32
//(4) GM: (N * OH) % 32
//(5) IC % 8 == 0
//(6) pw <= 3 && IW >= OW
//(7) OW >= 10
#ifndef COMPILE
#define CONV_3D_WINOGRAD_S16_W7_64X32R_P3_TEX
#endif
#ifndef CONV_3D_WINOGRAD_S16_W7_64X32R_P3_TEX
#define CONV_3D_WINOGRAD_S16_W7_64X32R_P3_TEX

template<int FH>
inline void conv3D_Winograd_s16_W7_64x32R_p3_tex(jlong* streams, int &index, int length,
	const float*  X, cudaTextureObject_t texX, int IH, int IW,
	const float* CW,//[FH, FW, IC, OC]
	      float*  Y, int OH, int OW,
	int N, int IC, int OC,
	int ph, int pw,
	int GN, int GM, int &GNr, int &GMr)
{
	//======[Stage1 (OW % 10): Winograd_F(10, 7) ]===========================================================
	next_cudaStream(stream, streams, index, length);
	conv3dWinograd_f10x7_k64x320R_p3_tex(stream, texX, IH, IW, CW, FH, Y, OH, OW, N, IC, OC, ph, pw);//common

	//======[Stage2 (OW % 4): Winograd_F(2 + 2, 7) ]==========================================================
	int OWr = OW % 10;
	if (OWr >= 4) {//Remainder: 4, 5, 6, 7, 8, 9
		const int ow_index = OW - OWr;
#ifdef ENBALE_CONV3D_WINOGRAD_FAX7R_CHANNEL_TEMPLATE
		//IC = 2^x
		if      (IC ==  64 && OC ==  64) conv3dWinograd_f2x7_ruse_k64x64RC_p3_CT_tex(stream, ow_index, texX, IH, IW, CW, FH, Y, OH, OW, N,  64,  64, ph, pw, OWr);
		else if (IC == 128 && OC == 128) conv3dWinograd_f2x7_ruse_k64x64RC_p3_CT_tex(stream, ow_index, texX, IH, IW, CW, FH, Y, OH, OW, N, 128, 128, ph, pw, OWr);
		else if (IC == 256 && OC == 256) conv3dWinograd_f2x7_ruse_k64x64RC_p3_CT_tex(stream, ow_index, texX, IH, IW, CW, FH, Y, OH, OW, N, 256, 256, ph, pw, OWr);
		else if (IC == 512 && OC == 512) conv3dWinograd_f2x7_ruse_k64x64RC_p3_CT_tex(stream, ow_index, texX, IH, IW, CW, FH, Y, OH, OW, N, 512, 512, ph, pw, OWr);
		//IC = 64x
		else if (IC == 192 && OC == 192) conv3dWinograd_f2x7_ruse_k64x64RC_p3_CT_tex(stream, ow_index, texX, IH, IW, CW, FH, Y, OH, OW, N, 192, 192, ph, pw, OWr);
		else if (IC == 320 && OC == 320) conv3dWinograd_f2x7_ruse_k64x64RC_p3_CT_tex(stream, ow_index, texX, IH, IW, CW, FH, Y, OH, OW, N, 320, 320, ph, pw, OWr);
		else if (IC == 384 && OC == 384) conv3dWinograd_f2x7_ruse_k64x64RC_p3_CT_tex(stream, ow_index, texX, IH, IW, CW, FH, Y, OH, OW, N, 384, 384, ph, pw, OWr);
		else if (IC == 448 && OC == 448) conv3dWinograd_f2x7_ruse_k64x64RC_p3_CT_tex(stream, ow_index, texX, IH, IW, CW, FH, Y, OH, OW, N, 448, 448, ph, pw, OWr);
		else
#endif
		conv3dWinograd_f2x7_ruse_k64x64RC_p3_tex(stream, ow_index, texX, IH, IW, CW, FH, Y, OH, OW, N, IC, OC, ph, pw, OWr);//common
		OWr = OWr & 3;//OWr % 4
	}

	//======[Stage3 (OW % 2): Winograd_F(2, 7) ]=============================================================
	if (OWr >= 2) {//Remainder: 2, 3
		const int ow_index = OW - OWr;
#ifdef ENBALE_CONV3D_WINOGRAD_FAX7R_CHANNEL_TEMPLATE
		//IC = 2^x
		if      (IC ==  64 && OC ==  64) conv3dWinograd_f2x7_k64x64RC_CT_tex(stream, ow_index, texX, IH, IW, CW, FH, Y, OH, OW, N,  64,  64, ph, pw, OWr);
		else if (IC == 128 && OC == 128) conv3dWinograd_f2x7_k64x64RC_CT_tex(stream, ow_index, texX, IH, IW, CW, FH, Y, OH, OW, N, 128, 128, ph, pw, OWr);
		else if (IC == 256 && OC == 256) conv3dWinograd_f2x7_k64x64RC_CT_tex(stream, ow_index, texX, IH, IW, CW, FH, Y, OH, OW, N, 256, 256, ph, pw, OWr);
		else if (IC == 512 && OC == 512) conv3dWinograd_f2x7_k64x64RC_CT_tex(stream, ow_index, texX, IH, IW, CW, FH, Y, OH, OW, N, 512, 512, ph, pw, OWr);
		//IC = 64x
		else if (IC == 192 && OC == 192) conv3dWinograd_f2x7_k64x64RC_CT_tex(stream, ow_index, texX, IH, IW, CW, FH, Y, OH, OW, N, 192, 192, ph, pw, OWr);
		else if (IC == 320 && OC == 320) conv3dWinograd_f2x7_k64x64RC_CT_tex(stream, ow_index, texX, IH, IW, CW, FH, Y, OH, OW, N, 320, 320, ph, pw, OWr);
		else if (IC == 448 && OC == 448) conv3dWinograd_f2x7_k64x64RC_CT_tex(stream, ow_index, texX, IH, IW, CW, FH, Y, OH, OW, N, 448, 448, ph, pw, OWr);
		else 
#endif
		conv3dWinograd_f2x7_k64x64RC_tex(stream, ow_index, texX, IH, IW, CW, FH, Y, OH, OW, N, IC, OC, ph, pw, OWr);//common
		OWr = OWr & 1;//OWr % 2
	}

	//======[Stage4: GEMM]===================================================================================
	if (OWr > 0) {//Remainder: 1
		const int GN64 = ((OC  ) >> 6 << 6);      //GN   = 64x
		const int GM32 = ((N*OH) >> 5 << 5) * OWr;//N*OH = 32x
		index = 0;//save L2 cache: texX ! = X
		conv3D_Gemm_32x32RC(streams, index, length,
			X, IH, IW, CW, FH, 7, Y, OH, OW, OWr, IC, OC, 1, 1, ph, pw,
			GN64, GM32, 0, 0, OW - OWr);
	}

	//=====[Process GNr and GMr]=============================================================================
	GNr = GN & 63;         //64 channels
	GMr = ((N*OH) & 31)*OW;//32 groups
}

#endif

#endif


//FW = 8
#ifndef CONV_3D_WINOGRAD_S16_W8_C32X32R_P4_TEXTURE
#define CONV_3D_WINOGRAD_S16_W8_C32X32R_P4_TEXTURE

#ifndef CONV_3D_WINOGRAD_S16_W8_C32X32R_P4_TEXTURE_BENCH_MARK_RTX3060TI
#define CONV_3D_WINOGRAD_S16_W8_C32X32R_P4_TEXTURE_BENCH_MARK_RTX3060TI

//[0] for: Feature = (128, 128), [N, IC, OC] = [32, 64, 64]
//LB = 4: Size = 128, Time = 10.7817 msec, Performace = 25494.9 GFlop/s
//WB = 4: Size = 128, Time = 10.1394 msec, Performace = 27109.9 GFlop/s
//cuDNN-NCHW-implicit-prec: Size = 128, Time = 16.749 msec, Performance = 16411.6 GFlop/s
//cuDNN-NHWC-implicit-prec: Size = 128, Time = 16.444 msec, Performance = 16716.0 GFlop/s

//[1] for: Feature = (112, 112), [N, IC, OC] = [32, 64, 64]
//LB = 4: Size = 98, Time = 8.65868 msec, Performace = 24305.5 GFlop/s
//WB = 4: Size = 98, Time = 8.16455 msec, Performace = 25776.5 GFlop/s
//cuDNN-NCHW-implicit-prec: Size = 98, Time = 12.95 msec, Performance = 16252.5 GFlop/s
//cuDNN-NHWC-implicit-prec: Size = 98, Time = 12.5  msec, Performance = 16836.3 GFlop/s

//[2] for: Feature = (72, 72), [N, IC, OC] = [64, 64, 64]
//LB = 4: Size = 81, Time = 6.51037 msec, Performace = 26718.3 GFlop/s
//WB = 4: Size = 81, Time = 6.13687 msec, Performace = 28344.4 GFlop/s
//cuDNN-NCHW-implicit-prec: Size = 81, Time = 10.952 msec, Performance = 15882.6 GFlop/s
//cuDNN-NHWC-implicit-prec: Size = 81, Time = 10.434 msec, Performance = 16671.1 GFlop/s

//[3] for: Feature = (64, 64), [N, IC, OC] = [128, 64, 64]
//LB = 4: Size = 128, Time = 10.7333 msec, Performace = 25609.7 GFlop/s
//WB = 4: Size = 128, Time = 10.0249 msec, Performace = 27419.4 GFlop/s
//cuDNN-NCHW-implicit-prec: Size = 128, Time = 16.858 msec, Performance = 16305.5 GFlop/s
//cuDNN-NHWC-implicit-prec: Size = 128, Time = 16.297 msec, Performance = 16866.8 GFlop/s

//[4] for: Feature = (56, 56), [N, IC, OC] = [128, 64, 64]
//LB = 4: Size = 98, Time = 8.63008 msec, Performace = 24386   GFlop/s
//WB = 4: Size = 98, Time = 8.13277 msec, Performace = 25877.2 GFlop/s
//cuDNN-NCHW-implicit-prec: Size = 98, Time = 13.069 msec, Performance = 16103.3 GFlop/s
//cuDNN-NHWC-implicit-prec: Size = 98, Time = 12.379 msec, Performance = 17000.8 GFlop/s

//[5] for: Feature = (36, 36), [N, IC, OC] = [64, 128, 128]
//LB = 4: Size = 81, Time = 7.44279 msec, Performace = 23371.1 GFlop/s
//WB = 4: Size = 81, Time = 6.79766 msec, Performace = 25589.1 GFlop/s
//cuDNN-NCHW-implicit-prec: Size = 81, Time = 10.773 msec, Performance = 16146.5 GFlop/s
//cuDNN-NHWC-implicit-prec: Size = 81, Time = 10.213 msec, Performance = 17031.8 GFlop/s

//[6] for: Feature = (32, 32), [N, IC, OC] = [128, 128, 128]
//LB = 4: Size = 128, Time = 12.5505 msec, Performace = 21901.8 GFlop/s
//WB = 4: Size = 128, Time = 11.7533 msec, Performace = 23387.2 GFlop/s
//cuDNN-NCHW-implicit-prec: Size = 128, Time = 15.95 msec, Performance = 17233.7 GFlop/s
//cuDNN-NHWC-implicit-prec: Size = 128, Time = 15.73 msec, Performance = 17474.8 GFlop/s

//[7] for: Feature = (28, 28), [N, IC, OC] = [128, 128, 128]
//LB = 4: Size = 98, Time = 10.1724 msec, Performace = 20688.6 GFlop/s
//WB = 4: Size = 98, Time = 9.67961 msec, Performace = 21741.9 GFlop/s
//cuDNN-NCHW-implicit-prec: Size = 98, Time = 12.474 msec, Performance = 16871.4 GFlop/s
//cuDNN-NHWC-implicit-prec: Size = 98, Time = 12.219 msec, Performance = 17223.5 GFlop/s

//[8] for: Feature = (18, 18), [N, IC, OC] = [64, 256, 256]
//LB = 4: Size = 81, Time = 7.64911 msec, Performace = 22740.7 GFlop/s
//WB = 4: Size = 81, Time = 7.28258 msec, Performace = 23885.2 GFlop/s
//cuDNN-NCHW-implicit-prec: Size = 81, Time = 10.82 msec, Performance = 16076.4 GFlop/s
//cuDNN-NHWC-implicit-prec: Size = 81, Time = 10.41 msec, Performance = 16707.9 GFlop/s

//[9] for: Feature = (9, 9), [N, IC, OC] = [64, 512, 512]
//LB = 4: Size = 81, Time = 7.4524  msec, Performace = 23341   GFlop/s
//WB = 4: Size = 81, Time = 6.82958 msec, Performace = 25469.5 GFlop/s
//cuDNN-NCHW-implicit-prec: Size = 81, Time = 10.715 msec, Performance = 16233.9 GFlop/s
//cuDNN-NHWC-implicit-prec: Size = 81, Time = 11.708 msec, Performance = 14857.0 GFlop/s

#endif

#ifndef CONV_3D_WINOGRAD_S16_W8_RUSE_C32X32R_P4_TEXTURE_BENCH_MARK_RTX3060TI
#define CONV_3D_WINOGRAD_S16_W8_RUSE_C32X32R_P4_TEXTURE_BENCH_MARK_RTX3060TI

//[0] for: Feature = (128, 128), [N, IC, OC] = [32, 64, 64]
//LB = 4: Size = 128, Time = 10.5173 msec, Performace = 26135.9 GFlop/s
//WB = 4: Size = 128, Time = 9.77328 msec, Performace = 28125.4 GFlop/s
//cuDNN-NCHW-implicit-prec: Size = 128, Time = 16.749 msec, Performance = 16411.6 GFlop/s
//cuDNN-NHWC-implicit-prec: Size = 128, Time = 16.444 msec, Performance = 16716.0 GFlop/s

//[1] for: Feature = (112, 112), [N, IC, OC] = [32, 64, 64] //108 + 3 + 1
//LB = 4: Size = 98, Time = 8.52326 msec, Performace = 24691.6 GFlop
//WB = 4: Size = 98, Time = 7.9419  msec, Performace = 26499.1 GFlop/s
//cuDNN-NCHW-implicit-prec: Size = 98, Time = 12.95 msec, Performance = 16252.5 GFlop/s
//cuDNN-NHWC-implicit-prec: Size = 98, Time = 12.5  msec, Performance = 16836.3 GFlop/s

//[2]for: Feature = (72, 72), [N, IC, OC] = [64, 64, 64]
//LB = 4: Size = 81, Time = 6.25068 msec, Performace = 27828.4 GFlop/s
//WB = 4: Size = 81, Time = 6.03845 msec, Performace = 28806.4 GFlop/s
//cuDNN-NCHW-implicit-prec: Size = 81, Time = 10.952 msec, Performance = 15882.6 GFlop/s
//cuDNN-NHWC-implicit-prec: Size = 81, Time = 10.434 msec, Performance = 16671.1 GFlop/s

//[3] for: Feature = (64, 64), [N, IC, OC] = [128, 64, 64] // 63 + 1
//LB = 4: Size = 128, Time = 10.8181 msec, Performace = 25409.1 GFlop/s
//WB = 4: Size = 128, Time = 10.0932 msec, Performace = 27234.1 GFlop/s
//cuDNN-NCHW-implicit-prec: Size = 128, Time = 16.858 msec, Performance = 16305.5/ GFlop/s
//cuDNN-NHWC-implicit-prec: Size = 128, Time = 16.297 msec, Performance = 16866.8 GFlop/s

//[4] for: Feature = (56, 56), [N, IC, OC] = [128, 64, 64]
//LB = 4: Size = 98, Time = 8.40832 msec, Performace = 25029.2 GFlop/s
//WB = 4: Size = 98, Time = 7.77221 msec, Performace = 27077.7 GFlop/s
//cuDNN-NCHW-implicit-prec: Size = 98, Time = 13.069 msec, Performance = 16103.3 GFlop/s
//cuDNN-NHWC-implicit-prec: Size = 98, Time = 12.379 msec, Performance = 17000.8 GFlop/s

//[5] for: Feature = (36, 36), [N, IC, OC] = [64, 128, 128]
//LB = 4: Size = 81, Time = 7.20521 msec, Performace = 24141.7 GFlop/s
//WB = 4: Size = 81, Time = 6.79766 msec, Performace = 25589.1 GFlop/s
//cuDNN-NCHW-implicit-prec: Size = 81, Time = 10.773 msec, Performance = 16146.5 GFlop/s
//cuDNN-NHWC-implicit-prec: Size = 81, Time = 10.213 msec, Performance = 17031.8 GFlop/s

//[6] for: Feature = (32, 32), [N, IC, OC] = [128, 128, 128] //27 + 5
//LB = 4: Size = 128, Time = 12.7675 msec, Performace = 21529.5 GFlop/s
//WB = 4: Size = 128, Time = 12.1775 msec, Performace = 22572.6 GFlop/s
//cuDNN-NCHW-implicit-prec: Size = 128, Time = 15.95 msec, Performance = 17233.7 GFlop/s
//cuDNN-NHWC-implicit-prec: Size = 128, Time = 15.73 msec, Performance = 17474.8 GFlop/s

//[7] for: Feature = (28, 28), [N, IC, OC] = [128, 128, 128]
//LB = 4: Size = 98, Time = 9.80561 msec, Performace = 21462.5 GFlop/s
//WB = 4: Size = 98, Time = 9.37882 msec, Performace = 22439.2 GFlop/s
//cuDNN-NCHW-implicit-prec: Size = 98, Time = 12.474 msec, Performance = 16871.4 GFlop/s
//cuDNN-NHWC-implicit-prec: Size = 98, Time = 12.219 msec, Performance = 17223.5 GFlop/s

//[8] for: Feature = (18, 18), [N, IC, OC] = [64, 256, 256]
//LB = 4: Size = 81, Time = 6.68368 msec, Performace = 26025.5 GFlop/s
//WB = 4: Size = 81, Time = 6.25848 msec, Performace = 27793.7 GFlop/s
//cuDNN-NCHW-implicit-prec: Size = 81, Time = 10.82 msec, Performance = 16076.4 GFlop/s
//cuDNN-NHWC-implicit-prec: Size = 81, Time = 10.41 msec, Performance = 16707.9 GFlop/s

//[9] for: Feature = (9, 9), [N, IC, OC] = [64, 512, 512]
//LB = 4: Size = 81, Time = 7.52293 msec, Performace = 23122.1 GFlop/s
//WB = 4: Size = 81, Time = 6.79719 msec, Performace = 25590.9 GFlop/s
//cuDNN-NCHW-implicit-prec: Size = 81, Time = 10.715 msec, Performance = 16233.9 GFlop/s
//cuDNN-NHWC-implicit-prec: Size = 81, Time = 11.708 msec, Performance = 14857.0 GFlop/s

#endif

//(1) FW = 8
//(2) sh = sw = 1
//(3) GN: (OC    ) % 32
//(4) GM: (N * OH) % 32
//(5) IC % 8 == 0
//(6) (pw <= 4) && (OW - IW - pw + 3 <= 0)
//(7) OW >= 9
#ifndef COMPILE
#define CONV_3D_WINOGRAD_S16_W8_C32X32R_P4_TEX
#endif
#ifndef CONV_3D_WINOGRAD_S16_W8_C32X32R_P4_TEX
#define CONV_3D_WINOGRAD_S16_W8_C32X32R_P4_TEX

template<int FH>
inline void conv3D_Winograd_s16_W8_32x32R_p4_tex(jlong* streams, int &index, int length,
	const float*  X, cudaTextureObject_t texX, int IH, int IW,
	const float* CW,//[FH, FW, IC, OC]
	      float*  Y, int OH, int OW,
	int N, int IC, int OC,
	int ph, int pw,
	int GN, int GM, int &GNr, int &GMr)
{
	//======[Stage1 (OW % 9) : F(9, 8) ]============================================================================
	next_cudaStream(stream, streams, index, length);
	if ((OW >= 18)) {
		conv3dWinograd_f9x8_ruse_k32x288R_p4_tex(stream, texX, IH, IW, CW, FH, Y, OH, OW, N, IC, OC, ph, pw);
		const int OWr = OW % 18;//align: OW % 9
		if (OWr >= 9) { const int ow_index = (OW - OWr);
#ifdef ENBALE_CONV3D_WINOGRAD_F9X8R_CHANNEL_TEMPLATE
			//IC = 2^x
			if      (IC ==  32 && OC ==  32) conv3dWinograd_f9x8_k32x288RC_p4_CT_tex(stream, ow_index, texX, IH, IW, CW, FH, Y, OH, OW, N,  32,  32, ph, pw, OWr);
			else if (IC ==  64 && OC ==  64) conv3dWinograd_f9x8_k32x288RC_p4_CT_tex(stream, ow_index, texX, IH, IW, CW, FH, Y, OH, OW, N,  64,  64, ph, pw, OWr);
			else if (IC == 128 && OC == 128) conv3dWinograd_f9x8_k32x288RC_p4_CT_tex(stream, ow_index, texX, IH, IW, CW, FH, Y, OH, OW, N, 128, 128, ph, pw, OWr);
			else if (IC == 256 && OC == 256) conv3dWinograd_f9x8_k32x288RC_p4_CT_tex(stream, ow_index, texX, IH, IW, CW, FH, Y, OH, OW, N, 256, 256, ph, pw, OWr);
			else if (IC == 512 && OC == 512) conv3dWinograd_f9x8_k32x288RC_p4_CT_tex(stream, ow_index, texX, IH, IW, CW, FH, Y, OH, OW, N, 512, 512, ph, pw, OWr);
			//IC = 64x
			else if (IC == 192 && OC == 192) conv3dWinograd_f9x8_k32x288RC_p4_CT_tex(stream, ow_index, texX, IH, IW, CW, FH, Y, OH, OW, N, 192, 192, ph, pw, OWr);
			else if (IC == 224 && OC == 224) conv3dWinograd_f9x8_k32x288RC_p4_CT_tex(stream, ow_index, texX, IH, IW, CW, FH, Y, OH, OW, N, 224, 224, ph, pw, OWr);
			else if (IC == 320 && OC == 320) conv3dWinograd_f9x8_k32x288RC_p4_CT_tex(stream, ow_index, texX, IH, IW, CW, FH, Y, OH, OW, N, 320, 320, ph, pw, OWr);
			else if (IC == 384 && OC == 384) conv3dWinograd_f9x8_k32x288RC_p4_CT_tex(stream, ow_index, texX, IH, IW, CW, FH, Y, OH, OW, N, 384, 384, ph, pw, OWr);
			else if (IC == 448 && OC == 448) conv3dWinograd_f9x8_k32x288RC_p4_CT_tex(stream, ow_index, texX, IH, IW, CW, FH, Y, OH, OW, N, 448, 448, ph, pw, OWr);
			else 
#endif
			conv3dWinograd_f9x8_k32x288RC_p4_tex(stream, ow_index, texX, IH, IW, CW, FH, Y, OH, OW, N, IC, OC, ph, pw, OWr);//common
		}
	}
#ifdef ENBALE_CONV3D_WINOGRAD_F9X8R_CHANNEL_TEMPLATE
	//IC = 2^x
	else if ((IC ==  32) && (OC ==  32)) conv3dWinograd_f9x8_k32x288R_p4_CT_tex(stream, texX, IH, IW, CW, FH, Y, OH, OW, N,  32,  32, ph, pw);
	else if ((IC ==  64) && (OC ==  64)) conv3dWinograd_f9x8_k32x288R_p4_CT_tex(stream, texX, IH, IW, CW, FH, Y, OH, OW, N,  64,  64, ph, pw);
	else if ((IC == 128) && (OC == 128)) conv3dWinograd_f9x8_k32x288R_p4_CT_tex(stream, texX, IH, IW, CW, FH, Y, OH, OW, N, 128, 128, ph, pw);
	else if ((IC == 256) && (OC == 256)) conv3dWinograd_f9x8_k32x288R_p4_CT_tex(stream, texX, IH, IW, CW, FH, Y, OH, OW, N, 256, 256, ph, pw);
	else if ((IC == 512) && (OC == 512)) conv3dWinograd_f9x8_k32x288R_p4_CT_tex(stream, texX, IH, IW, CW, FH, Y, OH, OW, N, 512, 512, ph, pw);
	//IC = 32x
	else if ((IC == 192) && (OC == 192)) conv3dWinograd_f9x8_k32x288R_p4_CT_tex(stream, texX, IH, IW, CW, FH, Y, OH, OW, N, 192, 192, ph, pw);
	else if ((IC == 224) && (OC == 224)) conv3dWinograd_f9x8_k32x288R_p4_CT_tex(stream, texX, IH, IW, CW, FH, Y, OH, OW, N, 224, 224, ph, pw);
	else if ((IC == 320) && (OC == 320)) conv3dWinograd_f9x8_k32x288R_p4_CT_tex(stream, texX, IH, IW, CW, FH, Y, OH, OW, N, 320, 320, ph, pw);
	else if ((IC == 384) && (OC == 384)) conv3dWinograd_f9x8_k32x288R_p4_CT_tex(stream, texX, IH, IW, CW, FH, Y, OH, OW, N, 384, 384, ph, pw);
	else if ((IC == 448) && (OC == 448)) conv3dWinograd_f9x8_k32x288R_p4_CT_tex(stream, texX, IH, IW, CW, FH, Y, OH, OW, N, 448, 448, ph, pw);
#endif
	else conv3dWinograd_f9x8_k32x288R_p4_tex(stream, texX, IH, IW, CW, FH, Y, OH, OW, N, IC, OC, ph, pw);//common

	//======[Stage2 (OW % 5): Winograd_F(5, 4)]=========================================================================
	int OWr = OW % 9;
	if ((OWr >= 5) && !(OC & 63)) {//Remainder: 5, 6, 7, 8
		conv3dWinograd_SFW_f5x4_k64x160RC_tex(stream, (OW - OWr), texX, IH, IW, CW, FH, 8, Y, OH, OW, N, IC, OC, ph, pw, OWr);
		OWr = OWr % 5;//OWr % 5
	}

	//======[Stage3 (OW % 3): Winograd_F(3, 2)]=========================================================================
	if ((OWr >= 3) && !((N*OH) & 63) && !(OC & 63)) {//Remainder: 3, 4
		conv3dWinograd_SFW_f3x2_k64x192RC_tex(stream, (OW - OWr), texX, IH, IW, CW, FH, 8, Y, OH, OW, N, IC, OC, ph, pw, OWr);
		OWr = OWr % 3;//OWr % 3
	}

	//======[Stage4: GEMM]==============================================================================================
	if (OWr > 0) {//Remainder: 1, 2
		const int GN32 = ((OC  ) >> 5 << 5);      //GN   = 32x
		const int GM32 = ((N*OH) >> 5 << 5) * OWr;//N*OH = 32x
		index = 0;//save L2 cache: texX != X
		conv3D_Gemm_32x32RC(streams, index, length,
			X, IH, IW, CW, FH, 8, Y, OH, OW, OWr, IC, OC, 1, 1, ph, pw,
			GN32, GM32, 0, 0, OW - OWr);
	}

	//=====[Process GNr and GMr]========================================================================================
	GNr = GN & 31;         //32 channels
	GMr = ((N*OH) & 31)*OW;//32 groups
}

#endif

#endif


//FW = 8
#ifndef CONV_3D_WINOGRAD_S16_W8_C64X32R_P4_TEXTURE
#define CONV_3D_WINOGRAD_S16_W8_C64X32R_P4_TEXTURE

#ifndef CONV_3D_WINOGRAD_S16_W8_C64X32R_P4_TEXTURE_BENCH_MARK_RTX3060TI
#define CONV_3D_WINOGRAD_S16_W8_C64X32R_P4_TEXTURE_BENCH_MARK_RTX3060TI

//[0] for: Feature = (128, 128), [N, IC, OC] = [32, 64, 64]
//LB = 4: Size = 128, Time = 9.56152 msec, Performace = 28748.3 GFlop/s
//WB = 4: Size = 128, Time = 8.74092 msec, Performace = 31447.2 GFlop/s
//cuDNN-NCHW-implicit-prec: Size = 128, Time = 16.749 msec, Performance = 16411.6 GFlop/s
//cuDNN-NHWC-implicit-prec: Size = 128, Time = 16.444 msec, Performance = 16716.0 GFlop/s

//[1] for: Feature = (112, 112), [N, IC, OC] = [32, 64, 64] //108 + 3 + 1
//LB = 4: Size = 98, Time = 7.89409 msec, Performace = 26659.6 GFlop/s
//WB = 4: Size = 98, Time = 7.7529  msec, Performace = 27145.1 GFlop/s
//cuDNN-NCHW-implicit-prec: Size = 98, Time = 12.95 msec, Performance = 16252.5 GFlop/s
//cuDNN-NHWC-implicit-prec: Size = 98, Time = 12.5  msec, Performance = 16836.3 GFlop/s

//[2]for: Feature = (72, 72), [N, IC, OC] = [64, 64, 64]
//LB = 4: Size = 81, Time = 5.58542 msec, Performace = 31142.9 GFlop/s
//WB = 4: Size = 81, Time = 5.08907 msec, Performace = 34180.4 GFlop/s
//cuDNN-NCHW-implicit-prec: Size = 81, Time = 10.952 msec, Performance = 15882.6 GFlop/s
//cuDNN-NHWC-implicit-prec: Size = 81, Time = 10.434 msec, Performance = 16671.1 GFlop/s

//[3] for: Feature = (64, 64), [N, IC, OC] = [128, 64, 64] // 63 + 1
//LB = 4: Size = 128, Time = 9.42227 msec, Performace = 29173.2 GFlop/s
//WB = 4: Size = 128, Time = 8.54895 msec, Performace = 32153.4 GFlop/s
//cuDNN-NCHW-implicit-prec: Size = 128, Time = 16.858 msec, Performance = 16305.5/ GFlop/s
//cuDNN-NHWC-implicit-prec: Size = 128, Time = 16.297 msec, Performance = 16866.8 GFlop/s

//[4] for: Feature = (56, 56), [N, IC, OC] = [128, 64, 64]
//LB = 4: Size = 98, Time = 7.53965 msec, Performace = 27912.9 GFlop/s
//WB = 4: Size = 98, Time = 6.96166 msec, Performace = 30230.3 GFlop/s
//cuDNN-NCHW-implicit-prec: Size = 98, Time = 13.069 msec, Performance = 16103.3 GFlop/s
//cuDNN-NHWC-implicit-prec: Size = 98, Time = 12.379 msec, Performance = 17000.8 GFlop/s

//[5] for: Feature = (36, 36), [N, IC, OC] = [64, 128, 128]
//LB = 4: Size = 81, Time = 6.44722 msec, Performace = 26980   GFlop/s
//WB = 4: Size = 81, Time = 6.04288 msec, Performace = 28785.3 GFlop/s
//cuDNN-NCHW-implicit-prec: Size = 81, Time = 10.773 msec, Performance = 16146.5 GFlop/s
//cuDNN-NHWC-implicit-prec: Size = 81, Time = 10.213 msec, Performance = 17031.8 GFlop/s

//[6] for: Feature = (32, 32), [N, IC, OC] = [128, 128, 128] //27 + 5
//LB = 4: Size = 128, Time = 10.9372 msec, Performace = 25132.3 GFlop/s
//WB = 4: Size = 128, Time = 10.3342 msec, Performace = 26598.8 GFlop/s
//cuDNN-NCHW-implicit-prec: Size = 128, Time = 15.95 msec, Performance = 17233.7 GFlop/s
//cuDNN-NHWC-implicit-prec: Size = 128, Time = 15.73 msec, Performance = 17474.8 GFlop/s

//[7] for: Feature = (28, 28), [N, IC, OC] = [128, 128, 128]
//LB = 4: Size = 98, Time = 8.33407 msec, Performace = 25252.2 GFlop/s
//WB = 4: Size = 98, Time = 7.72568 msec, Performace = 27240.8 GFlop/s
//cuDNN-NCHW-implicit-prec: Size = 98, Time = 12.474 msec, Performance = 16871.4 GFlop/s
//cuDNN-NHWC-implicit-prec: Size = 98, Time = 12.219 msec, Performance = 17223.5 GFlop/s

//[8] for: Feature = (18, 18), [N, IC, OC] = [64, 256, 256]
//LB = 4: Size = 81, Time = 6.08407 msec, Performace = 28590.4 GFlop/s
//WB = 4: Size = 81, Time = 5.66385 msec, Performace = 30711.6 GFlop/s
//cuDNN-NCHW-implicit-prec: Size = 81, Time = 10.82 msec, Performance = 16076.4 GFlop/s
//cuDNN-NHWC-implicit-prec: Size = 81, Time = 10.41 msec, Performance = 16707.9 GFlop/s

//[9] for: Feature = (9, 9), [N, IC, OC] = [64, 512, 512]
//LB = 4: Size = 81, Time = 5.90664 msec, Performace = 29449.3 GFlop/s
//WB = 4: Size = 81, Time = 5.24217 msec, Performace = 33182.1 GFlop/s
//cuDNN-NCHW-implicit-prec: Size = 81, Time = 10.715 msec, Performance = 16233.9 GFlop/s
//cuDNN-NHWC-implicit-prec: Size = 81, Time = 11.708 msec, Performance = 14857.0 GFlop/s

#endif

//(1) FW = 8
//(2) sh = sw = 1
//(3) GN: (OC    ) % 64
//(4) GM: (N * OH) % 32
//(5) IC % 8 == 0
//(6) (pw <= 4) && (OW - IW - pw + 3 <= 0)
//(7) OW >= 9
#ifndef COMPILE
#define CONV_3D_WINOGRAD_S16_W8_C64X32R_P4_TEX
#endif
#ifndef CONV_3D_WINOGRAD_S16_W8_C64X32R_P4_TEX
#define CONV_3D_WINOGRAD_S16_W8_C64X32R_P4_TEX

template<int FH>
inline void conv3D_Winograd_s16_W8_64x32R_p4_tex(jlong* streams, int &index, int length,
	const float*  X, cudaTextureObject_t texX, int IH, int IW,
	const float* CW,//[FH, FW, IC, OC]
	      float*  Y, int OH, int OW,
	int N, int IC, int OC,
	int ph, int pw,
	int GN, int GM, int &GNr, int &GMr)
{
	//======[Stage1 (OW % 9) : F(9, 8) ]============================================================================
	next_cudaStream(stream, streams, index, length);
	conv3dWinograd_f9x8_k64x288R_p4_tex(stream, texX, IH, IW, CW, FH, Y, OH, OW, N, IC, OC, ph, pw);//common

	//======[Stage2 (OW % 5): Winograd_F(5, 4)]=========================================================================
	int OWr = OW % 9;
	if (OWr >= 5) {//Remainder: 5, 6, 7, 8
		conv3dWinograd_SFW_f5x4_k64x160RC_tex(stream, (OW - OWr), texX, IH, IW, CW, FH, 8, Y, OH, OW, N, IC, OC, ph, pw, OWr);
		OWr = OWr % 5;
	}

	//======[Stage3 (OW % 3): Winograd_F(3, 2)]=========================================================================
	if ((OWr >= 3) && !((N*OH) & 63)) {//Remainder: 3, 4
		conv3dWinograd_SFW_f3x2_k64x192RC_tex(stream, (OW - OWr), texX, IH, IW, CW, FH, 8, Y, OH, OW, N, IC, OC, ph, pw, OWr);
		OWr = OWr % 3;
	}

	//======[Stage4: GEMM]==============================================================================================
	if (OWr > 0) {//Remainder: 1, 2
		const int GN64 = ((OC  ) >> 6 << 6);      //GN   = 64x
		const int GM32 = ((N*OH) >> 5 << 5) * OWr;//N*OH = 32x
		index = 0;//save L2 cache: texX != X
		conv3D_Gemm_32x32RC(streams, index, length,
			X, IH, IW, CW, FH, 8, Y, OH, OW, OWr, IC, OC, 1, 1, ph, pw,
			GN64, GM32, 0, 0, OW - OWr);
	}

	//=====[Process GNr and GMr]========================================================================================
	GNr = GN & 63;         //64 channels
	GMr = ((N*OH) & 31)*OW;//32 groups
}

#endif

#endif


//FW = 9
#ifndef CONV_3D_WINOGRAD_S16_W9_C32X32R_P4_TEXTURE
#define CONV_3D_WINOGRAD_S16_W9_C32X32R_P4_TEXTURE

#ifndef CONV_3D_WINOGRAD_S16_W9_C32X32R_P4_TEXTURE_BENCH_MARK_RTX3060TI
#define CONV_3D_WINOGRAD_S16_W9_C32X32R_P4_TEXTURE_BENCH_MARK_RTX3060TI

//[0] for: Feature = (128, 128), [N, IC, OC] = [32, 64, 64]
//LB = 4: Size = 162, Time = 12.9847 msec, Performace = 26792.5 GFlop/s
//WB = 4: Size = 162, Time = 12.2418 msec, Performace = 28418.4 GFlop/s
//cuDNN-NCHW-implicit-prec: Size = 162, Time = 20.75 msec, Performance = 16765.9 GFlop/s
//cuDNN-NHWC-implicit-prec: Size = 162, Time = 20.15 msec, Performance = 17265.1 GFlop/s
//cuDNN-NHWC-GEMM-implicit: Size = 162, Time = 106.91 msec, Performance = 3254.1 GFlop/s

//[1] for: Feature = (124, 124), [N, IC, OC] = [32, 64, 64]
//LB = 4: Size = 152.033203, Time = 12.9784 msec, Performace = 25156.2 GFlop/s
//WB = 4: Size = 152.033203, Time = 12.3133 msec, Performace = 26515.1 GFlop/s
//cuDNN-NCHW-implicit-prec: Size = 152.033203, Time = 19.61 msec, Performance = 16649.1 GFlop/s
//cuDNN-NHWC-implicit-prec: Size = 152.033203, Time = 19.07 msec, Performance = 17120.5 GFlop/s
//cuDNN-NHWC-GEMM-implicit: Size = 152.033203, Time = 100.87 msec, Performance = 3236.7 GFlop/s

//[2] for: Feature = (96, 96), [N, IC, OC] = [32, 64, 64]
//LB = 4: Size = 91.125, Time = 7.46563 msec, Performace = 26212   GFlop/s
//WB = 4: Size = 91.125, Time = 7.03347 msec, Performace = 27822.6 GFlop/s
//cuDNN-NCHW-implicit-prec: Size = 91.125, Time = 11.95  msec, Performance = 16375.7 GFlop/s
//cuDNN-NHWC-implicit-prec: Size = 91.125, Time = 11.465 msec, Performance = 17068.4 GFlop/s
//cuDNN-NHWC-GEMM-implicit: Size = 91.125, Time = 59.84 msec, Performance = 3270.2 GFlop/s

//[3] for: Feature = (64, 64), [N, IC, OC] = [128, 64, 64]
//LB = 4: Size = 162, Time = 13.9629 msec, Performace = 24915.5 GFlop/s
//WB = 4: Size = 162, Time = 12.0937 msec, Performace = 28766.4 GFlop/s
//cuDNN-NCHW-implicit-prec: Size = 162, Time = 20.805 msec, Performance = 16721.6 GFlop/s
//cuDNN-NHWC-implicit-prec: Size = 162, Time = 20.245 msec, Performance = 17184.1 GFlop/s
//cuDNN-NHWC-GEMM-implicit: Size = 162, Time = 106.27 msec, Performance = 3273.7 GFlop/s

//[4] for: Feature = (60, 60), [N, IC, OC] = [128, 64, 64]
//LB = 4: Size = 142.383813, Time = 12.5925 msec, Performace = 24281.5 GFlop/s
//WB = 4: Size = 142.383813, Time = 11.7471 msec, Performace = 26029 GFlop/s
//cuDNN-NCHW-implicit-prec: Size = 142.382813, Time = 18.475 msec, Performance = 16550.2 GFlop/s
//cuDNN-NHWC-implicit-prec: Size = 142.382813, Time = 17.735 msec, Performance = 17240.8 GFlop/s
//cuDNN-NHWC-GEMM-implicit: Size = 142.382813, Time = 92.565 msec, Performance = 3303.2 GFlop/s

//[5] for: Feature = (48, 48), [N, IC, OC] = [128, 64, 64]
//LB = 4: Size = 91.125, Time = 7.39469 msec, Performace = 26463.5 GFlop/s
//WB = 4: Size = 91.125, Time = 7.01216 msec, Performace = 27907.1 GFlop/s
//cuDNN-NCHW-implicit-prec: Size = 91.125, Time = 11.98 msec, Performance = 16334.7 GFlop/s
//cuDNN-NHWC-implicit-prec: Size = 91.125, Time = 11.59 msec, Performance = 16884.3 GFlop/s
//cuDNN-NHWC-GEMM-implicit: Size = 91.125, Time = 59.015 msec, Performance = 3315.9 GFlop/s

//[6] for: Feature = (32, 32), [N, IC, OC] = [128, 128, 128]
//LB = 4: Size = 162, Time = 15.0487 msec, Performace = 23117.7 GFlop/s
//WB = 4: Size = 162, Time = 14.0815 msec, Performace = 24705.7 GFlop/s
//cuDNN-NCHW-implicit-prec: Size = 162, Time = 20.29 msec, Performance = 17146.0 GFlop/s
//cuDNN-NHWC-implicit-prec: Size = 162, Time = 19.6  msec, Performance = 17749.6 GFlop/s
//cuDNN-NHWC-GEMM-implicit: Size = 162, Time = 55.865 msec, Performance = 6227.4 GFlop/s

//[7] for: Feature = (28, 28), [N, IC, OC] = [128, 128, 128]
//LB = 4: Size = 124.03125, Time = 13.6618 msec, Performace = 19496.4 GFlop/s
//WB = 4: Size = 124.03125, Time = 13.0162 msec, Performace = 20463.4 GFlop/s
//cuDNN-NCHW-implicit-prec: Size = 124.03125, Time = 15.69  msec, Performance = 16976.1 GFlop/s
//cuDNN-NHWC-implicit-prec: Size = 124.03125, Time = 15.345 msec, Performance = 17357.8 GFlop/s
//cuDNN-NHWC-GEMM-implicit: Size = 124.03125, Time = 42.635 msec, Performance = 6247.3 GFlop/s

//[8] for: Feature = (16, 16), [N, IC, OC] = [128, 256, 256]
//LB = 4: Size = 162, Time = 14.33   msec, Performace = 24277.1 GFlop/s
//WB = 4: Size = 162, Time = 13.5423 msec, Performace = 25689.4 GFlop/s
//cuDNN-NCHW-implicit-prec: Size = 162, Time = 19.675 msec, Performance = 17681.9 GFlop/s
//cuDNN-NHWC-implicit-prec: Size = 162, Time = 20.29  msec, Performance = 17146.0 GFlop/s
//cuDNN-NHWC-GEMM-implicit: Size = 162, Time = 62.395 msec, Performance = 5575.6 GFlop/s

//[9] for: Feature = (8, 8), [N, IC, OC] = [128, 512, 512]
//LB = 4: Size = 162, Time = 14.3395 msec, Performace = 24261.2 GFlop/s
//WB = 4: Size = 162, Time = 13.4041 msec, Performace = 25954.1 GFlop/s
//cuDNN-NCHW-implicit-prec: Size = 162, Time = 20.435 msec, Performance = 17024.3 GFlop/s
//cuDNN-NHWC-implicit-prec: Size = 162, Time = 20.1   msec, Performance = 17308.1 GFlop/s
//cuDNN-NHWC-GEMM-implicit: Size = 162, Time = 51.98 msec, Performance = 6692.8 GFlop/s

#endif

#ifndef CONV_3D_WINOGRAD_S16_W9_RUSE_C32X32R_P4_TEXTURE_BENCH_MARK_RTX3060TI
#define CONV_3D_WINOGRAD_S16_W9_RUSE_C32X32R_P4_TEXTURE_BENCH_MARK_RTX3060TI

//[0] for: Feature = (128, 128), [N, IC, OC] = [32, 64, 64]
//LB = 4: Size = 162, Time = 12.56   msec, Performace = 27698.5 GFlop/s
//WB = 4: Size = 162, Time = 11.4723 msec, Performace = 30324.4 GFlop/s
//cuDNN-NCHW-implicit-prec: Size = 162, Time = 20.75 msec, Performance = 16765.9 GFlop/s
//cuDNN-NHWC-implicit-prec: Size = 162, Time = 20.15 msec, Performance = 17265.1 GFlop/s
//cuDNN-NHWC-GEMM-implicit: Size = 162, Time = 106.91 msec, Performance = 3254.1 GFlop/s

//[1] for: Feature = (124, 124), [N, IC, OC] = [32, 64, 64]
//LB = 4: Size = 152.033203, Time = 12.9863 msec, Performace = 25141   GFlop/s
//WB = 4: Size = 152.033203, Time = 12.1006 msec, Performace = 26981.1 GFlop/s
//cuDNN-NCHW-implicit-prec: Size = 152.033203, Time = 19.61 msec, Performance = 16649.1 GFlop/s
//cuDNN-NHWC-implicit-prec: Size = 152.033203, Time = 19.07 msec, Performance = 17120.5 GFlop/s
//cuDNN-NHWC-GEMM-implicit: Size = 152.033203, Time = 100.87 msec, Performance = 3236.7 GFlop/s

//[2] for: Feature = (96, 96), [N, IC, OC] = [32, 64, 64]
//LB = 4: Size = 91.125, Time = 6.95215 msec, Performace = 28148.1 GFlop/s
//WB = 4: Size = 91.125, Time = 6.44336 msec, Performace = 30370.7 GFlop/s
//cuDNN-NCHW-implicit-prec: Size = 91.125, Time = 11.95  msec, Performance = 16375.7 GFlop/s
//cuDNN-NHWC-implicit-prec: Size = 91.125, Time = 11.465 msec, Performance = 17068.4 GFlop/s
//cuDNN-NHWC-GEMM-implicit: Size = 91.125, Time = 59.84 msec, Performance = 3270.2 GFlop/s

//[3] for: Feature = (64, 64), [N, IC, OC] = [128, 64, 64]
//LB = 4: Size = 162, Time = 12.2029 msec, Performace = 28509   GFlop/s
//WB = 4: Size = 162, Time = 11.4095 msec, Performace = 30491.5 GFlop/s
//cuDNN-NCHW-implicit-prec: Size = 162, Time = 20.805 msec, Performance = 16721.6 GFlop/s
//cuDNN-NHWC-implicit-prec: Size = 162, Time = 20.245 msec, Performance = 17184.1 GFlop/s
//cuDNN-NHWC-GEMM-implicit: Size = 162, Time = 106.27 msec, Performance = 3273.7 GFlop/s

//[4] for: Feature = (60, 60), [N, IC, OC] = [128, 64, 64]
//LB = 4: Size = 142.383813, Time = 12.9701 msec, Performace = 23574.6 GFlop/s
//WB = 4: Size = 142.383813, Time = 12.1358 msec, Performace = 25195.2 GFlop/s
//cuDNN-NCHW-implicit-prec: Size = 142.382813, Time = 18.475 msec, Performance = 16550.2 GFlop/s
//cuDNN-NHWC-implicit-prec: Size = 142.382813, Time = 17.735 msec, Performance = 17240.8 GFlop/s
//cuDNN-NHWC-GEMM-implicit: Size = 142.382813, Time = 92.565 msec, Performance = 3303.2 GFlop/s

//[5] for: Feature = (48, 48), [N, IC, OC] = [128, 64, 64]
//LB = 4: Size = 91.125, Time = 6.78992 msec, Performace = 28820.6 GFlop/s
//WB = 4: Size = 91.125, Time = 6.3524  msec, Performace = 30805.6 GFlop/s
//cuDNN-NCHW-implicit-prec: Size = 91.125, Time = 11.98 msec, Performance = 16334.7 GFlop/s
//cuDNN-NHWC-implicit-prec: Size = 91.125, Time = 11.59 msec, Performance = 16884.3 GFlop/s
//cuDNN-NHWC-GEMM-implicit: Size = 91.125, Time = 59.015 msec, Performance = 3315.9 GFlop/s

//[6] for: Feature = (32, 32), [N, IC, OC] = [128, 128, 128]
//LB = 4: Size = 162, Time = 13.771  msec, Performace = 25262.7 GFlop/s
//WB = 4: Size = 162, Time = 13.1663 msec, Performace = 26422.9 GFlop/s
//cuDNN-NCHW-implicit-prec: Size = 162, Time = 20.29 msec, Performance = 17146.0 GFlop/s
//cuDNN-NHWC-implicit-prec: Size = 162, Time = 19.6  msec, Performance = 17749.6 GFlop/s
//cuDNN-NHWC-GEMM-implicit: Size = 162, Time = 55.865 msec, Performance = 6227.4 GFlop/s

//[7] for: Feature = (28, 28), [N, IC, OC] = [128, 128, 128]
//LB = 4: Size = 124.03125, Time = 13.1492 msec, Performace = 20256.4 GFlop/s
//WB = 4: Size = 124.03125, Time = 12.6588 msec, Performace = 21041.1 GFlop/s
//cuDNN-NCHW-implicit-prec: Size = 124.03125, Time = 15.69  msec, Performance = 16976.1 GFlop/s
//cuDNN-NHWC-implicit-prec: Size = 124.03125, Time = 15.345 msec, Performance = 17357.8 GFlop/s
//cuDNN-NHWC-GEMM-implicit: Size = 124.03125, Time = 42.635 msec, Performance = 6247.3 GFlop/s

//[8] for: Feature = (16, 16), [N, IC, OC] = [128, 256, 256]
//LB = 4: Size = 162, Time = 12.557  msec, Performace = 27705   GFlop/s
//WB = 4: Size = 162, Time = 11.7325 msec, Performace = 29652.1 GFlop/s
//cuDNN-NCHW-implicit-prec: Size = 162, Time = 19.675 msec, Performance = 17681.9 GFlop/s
//cuDNN-NHWC-implicit-prec: Size = 162, Time = 20.29  msec, Performance = 17146.0 GFlop/s
//cuDNN-NHWC-GEMM-implicit: Size = 162, Time = 62.395 msec, Performance = 5575.6 GFlop/s

#endif

//(1) FW = 9
//(2) sh = sw = 1
//(3) GN: (OC    ) % 32
//(4) GM: (N * OH) % 32
//(5) IC % 8 == 0
//(6) pw <= 4) && (OW - IW - pw + 4 <= 0)
//(7) OW >= 8
#ifndef COMPILE
#define CONV_3D_WINOGRAD_S16_W9_C32X32R_P4_TEX
#endif
#ifndef CONV_3D_WINOGRAD_S16_W9_C32X32R_P4_TEX
#define CONV_3D_WINOGRAD_S16_W9_C32X32R_P4_TEX

template<int FH>
inline void conv3D_Winograd_s16_W9_32x32R_p4_tex(jlong* streams, int &index, int length,
	const float*  X, cudaTextureObject_t texX, int IH, int IW,
	const float* CW,//[FH, FW, IC, OC]
	      float*  Y, int OH, int OW,
	int N, int IC, int OC,
	int ph, int pw,
	int GN, int GM, int &GNr, int &GMr)
{
	//======[Stage1 (OW % 8) : F(8, 9) ]============================================================================
	next_cudaStream(stream, streams, index, length);
	if ((OW >= 16)) {
		conv3dWinograd_f8x9_ruse_k32x256R_p4_tex(stream, texX, IH, IW, CW, FH, Y, OH, OW, N, IC, OC, ph, pw);
		const int OWr = OW & 15;//align: OW % 8
		if (OWr >= 8) { const int ow_index = (OW - OWr);
#ifdef ENBALE_CONV3D_WINOGRAD_F8X9R_CHANNEL_TEMPLATE
			//IC = 2^x
			if      (IC ==  32 && OC ==  32) conv3dWinograd_f8x9_k32x256RC_p4_CT_tex(stream, ow_index, texX, IH, IW, CW, FH, Y, OH, OW, N,  32,  32, ph, pw, OWr);
			else if (IC ==  64 && OC ==  64) conv3dWinograd_f8x9_k32x256RC_p4_CT_tex(stream, ow_index, texX, IH, IW, CW, FH, Y, OH, OW, N,  64,  64, ph, pw, OWr);
			else if (IC == 128 && OC == 128) conv3dWinograd_f8x9_k32x256RC_p4_CT_tex(stream, ow_index, texX, IH, IW, CW, FH, Y, OH, OW, N, 128, 128, ph, pw, OWr);
			else if (IC == 256 && OC == 256) conv3dWinograd_f8x9_k32x256RC_p4_CT_tex(stream, ow_index, texX, IH, IW, CW, FH, Y, OH, OW, N, 256, 256, ph, pw, OWr);
			else if (IC == 512 && OC == 512) conv3dWinograd_f8x9_k32x256RC_p4_CT_tex(stream, ow_index, texX, IH, IW, CW, FH, Y, OH, OW, N, 512, 512, ph, pw, OWr);
			//IC = 32x
			else if (IC == 192 && OC == 192) conv3dWinograd_f8x9_k32x256RC_p4_CT_tex(stream, ow_index, texX, IH, IW, CW, FH, Y, OH, OW, N, 192, 192, ph, pw, OWr);
			else if (IC == 224 && OC == 224) conv3dWinograd_f8x9_k32x256RC_p4_CT_tex(stream, ow_index, texX, IH, IW, CW, FH, Y, OH, OW, N, 224, 224, ph, pw, OWr);
			else if (IC == 320 && OC == 320) conv3dWinograd_f8x9_k32x256RC_p4_CT_tex(stream, ow_index, texX, IH, IW, CW, FH, Y, OH, OW, N, 320, 320, ph, pw, OWr);
			else if (IC == 384 && OC == 384) conv3dWinograd_f8x9_k32x256RC_p4_CT_tex(stream, ow_index, texX, IH, IW, CW, FH, Y, OH, OW, N, 384, 384, ph, pw, OWr);
			else if (IC == 448 && OC == 448) conv3dWinograd_f8x9_k32x256RC_p4_CT_tex(stream, ow_index, texX, IH, IW, CW, FH, Y, OH, OW, N, 448, 448, ph, pw, OWr);
			else 
#endif
			conv3dWinograd_f8x9_k32x256RC_p4_tex(stream, ow_index, texX, IH, IW, CW, FH, Y, OH, OW, N, IC, OC, ph, pw, OWr);//common
		}
	}
#ifdef ENBALE_CONV3D_WINOGRAD_F8X9R_CHANNEL_TEMPLATE
	//IC = 2^x
	else if (IC ==  32) { if (OC ==  32) conv3dWinograd_f8x9_k32x256R_p4_CT_tex(stream, texX, IH, IW, CW, FH, Y, OH, OW, N,  32,  32, ph, pw); else conv3dWinograd_f8x9_k32x256R_p4_ICT_tex(stream, texX, IH, IW, CW, FH, Y, OH, OW, N,  32, OC, ph, pw); }
	else if (IC ==  64) { if (OC ==  64) conv3dWinograd_f8x9_k32x256R_p4_CT_tex(stream, texX, IH, IW, CW, FH, Y, OH, OW, N,  64,  64, ph, pw); else conv3dWinograd_f8x9_k32x256R_p4_ICT_tex(stream, texX, IH, IW, CW, FH, Y, OH, OW, N,  64, OC, ph, pw); }
	else if (IC == 128) { if (OC == 128) conv3dWinograd_f8x9_k32x256R_p4_CT_tex(stream, texX, IH, IW, CW, FH, Y, OH, OW, N, 128, 128, ph, pw); else conv3dWinograd_f8x9_k32x256R_p4_ICT_tex(stream, texX, IH, IW, CW, FH, Y, OH, OW, N, 128, OC, ph, pw); }
	else if (IC == 256) { if (OC == 256) conv3dWinograd_f8x9_k32x256R_p4_CT_tex(stream, texX, IH, IW, CW, FH, Y, OH, OW, N, 256, 256, ph, pw); else conv3dWinograd_f8x9_k32x256R_p4_ICT_tex(stream, texX, IH, IW, CW, FH, Y, OH, OW, N, 256, OC, ph, pw); }
	else if (IC == 512) { if (OC == 512) conv3dWinograd_f8x9_k32x256R_p4_CT_tex(stream, texX, IH, IW, CW, FH, Y, OH, OW, N, 512, 512, ph, pw); else conv3dWinograd_f8x9_k32x256R_p4_ICT_tex(stream, texX, IH, IW, CW, FH, Y, OH, OW, N, 512, OC, ph, pw); }
	//IC = 32x
	else if (IC == 192) { if (OC == 192) conv3dWinograd_f8x9_k32x256R_p4_CT_tex(stream, texX, IH, IW, CW, FH, Y, OH, OW, N, 192, 192, ph, pw); else conv3dWinograd_f8x9_k32x256R_p4_ICT_tex(stream, texX, IH, IW, CW, FH, Y, OH, OW, N, 192, OC, ph, pw); }
	else if (IC == 224) { if (OC == 224) conv3dWinograd_f8x9_k32x256R_p4_CT_tex(stream, texX, IH, IW, CW, FH, Y, OH, OW, N, 224, 224, ph, pw); else conv3dWinograd_f8x9_k32x256R_p4_ICT_tex(stream, texX, IH, IW, CW, FH, Y, OH, OW, N, 224, OC, ph, pw); }
	else if (IC == 320) { if (OC == 320) conv3dWinograd_f8x9_k32x256R_p4_CT_tex(stream, texX, IH, IW, CW, FH, Y, OH, OW, N, 320, 320, ph, pw); else conv3dWinograd_f8x9_k32x256R_p4_ICT_tex(stream, texX, IH, IW, CW, FH, Y, OH, OW, N, 320, OC, ph, pw); }
	else if (IC == 384) { if (OC == 384) conv3dWinograd_f8x9_k32x256R_p4_CT_tex(stream, texX, IH, IW, CW, FH, Y, OH, OW, N, 384, 384, ph, pw); else conv3dWinograd_f8x9_k32x256R_p4_ICT_tex(stream, texX, IH, IW, CW, FH, Y, OH, OW, N, 384, OC, ph, pw); }
	else if (IC == 448) { if (OC == 448) conv3dWinograd_f8x9_k32x256R_p4_CT_tex(stream, texX, IH, IW, CW, FH, Y, OH, OW, N, 448, 448, ph, pw); else conv3dWinograd_f8x9_k32x256R_p4_ICT_tex(stream, texX, IH, IW, CW, FH, Y, OH, OW, N, 448, OC, ph, pw); }
#endif
	else conv3dWinograd_f8x9_k32x256R_p4_tex(stream, texX, IH, IW, CW, FH, Y, OH, OW, N, IC, OC, ph, pw);//common

	//======[Stage2 (OW % 4): Winograd_F(2+2, 3)]=======================================================================
	int OWr = OW & 7;
	if ((OWr >= 4) && !((N*OH) & 31) && !(OC & 63)) {//Remainder: 4, 5, 6, 7, 8
		conv3dWinograd_SFW_f2x3_k64x128R4C_tex(stream, (OW - OWr), texX, IH, IW, CW, FH, 9, Y, OH, OW, N, IC, OC, ph, pw, OWr);
		OWr = OWr & 3;//OWr % 4
	}

	//======[Stage3 (OW % 2): Winograd_F(2, 3)]=========================================================================
	if ((OWr >= 2) && !((N*OH) & 63) && !(OC & 63)) {//Remainder: 2, 3
		conv3dWinograd_SFW_f2x3_k64x128RC_tex(stream, (OW - OWr), texX, IH, IW, CW, FH, 9, Y, OH, OW, N, IC, OC, ph, pw, OWr);
		OWr = OWr & 1;//OWr % 2
	}

	//======[Stage4: GEMM]==============================================================================================
	if (OWr > 0) {//Remainder: 1
		const int GN32 = ((OC  ) >> 5 << 5);      //GN   = 32x
		const int GM32 = ((N*OH) >> 5 << 5) * OWr;//N*OH = 32x
		index = 0;//save L2 cache: texX != X
		conv3D_Gemm_32x32RC(streams, index, length,
			X, IH, IW, CW, FH, 9, Y, OH, OW, OWr, IC, OC, 1, 1, ph, pw,
			GN32, GM32, 0, 0, OW - OWr);
	}

	//=====[Process GNr and GMr]========================================================================================
	GNr = GN & 31;//32 channels
	GMr = ((N*OH) & 31)*OW;//32 groups
}

#endif

#endif


//FW = 9
#ifndef CONV_3D_WINOGRAD_S16_W9_C64X32R_P4_TEXTURE
#define CONV_3D_WINOGRAD_S16_W9_C64X32R_P4_TEXTURE

#ifndef CONV_3D_WINOGRAD_S16_W9_C64X32R_P4_TEXTURE_BENCH_MARK_RTX3060TI
#define CONV_3D_WINOGRAD_S16_W9_C64X32R_P4_TEXTURE_BENCH_MARK_RTX3060TI

//[0] for: Feature = (128, 128), [N, IC, OC] = [32, 64, 64]
//LB = 4: Size = 162, Time = 11.2419 msec, Performace = 30946   GFlop/s
//WB = 4: Size = 162, Time = 10.351  msec, Performace = 33609.5 GFlop/s
//cuDNN-NCHW-implicit-prec: Size = 162, Time = 20.75 msec, Performance = 16765.9 GFlop/s
//cuDNN-NHWC-implicit-prec: Size = 162, Time = 20.15 msec, Performance = 17265.1 GFlop/s
//cuDNN-NHWC-GEMM-implicit: Size = 162, Time = 106.91 msec, Performance = 3254.1 GFlop/s

//[1] for: Feature = (124, 124), [N, IC, OC] = [32, 64, 64]
//LB = 4: Size = 152.033, Time = 11.2726 msec, Performace = 28963   GFlop/s
//WB = 4: Size = 152.033, Time = 10.5374 msec, Performace = 30983.7 GFlop/s
//cuDNN-NCHW-implicit-prec: Size = 152.033203, Time = 19.61 msec, Performance = 16649.1 GFlop/s
//cuDNN-NHWC-implicit-prec: Size = 152.033203, Time = 19.07 msec, Performance = 17120.5 GFlop/s
//cuDNN-NHWC-GEMM-implicit: Size = 152.033203, Time = 100.87 msec, Performance = 3236.7 GFlop/s

//[2] for: Feature = (96, 96), [N, IC, OC] = [32, 64, 64]
//LB = 4: Size = 91.125, Time = 6.4298  msec, Performace = 30434.7 GFlop/s
//WB = 4: Size = 91.125, Time = 5.94506 msec, Performace = 32916.3 GFlop/s
//cuDNN-NCHW-implicit-prec: Size = 91.125, Time = 11.95  msec, Performance = 16375.7 GFlop/s
//cuDNN-NHWC-implicit-prec: Size = 91.125, Time = 11.465 msec, Performance = 17068.4 GFlop/s
//cuDNN-NHWC-GEMM-implicit: Size = 91.125, Time = 59.84 msec, Performance = 3270.2 GFlop/s

//[3] for: Feature = (64, 64), [N, IC, OC] = [128, 64, 64]
//LB = 4: Size = 162, Time = 11.1499 msec, Performace = 31201.3 GFlop/s
//WB = 4: Size = 162, Time = 10.2437 msec, Performace = 33961.5 GFlop/s
//cuDNN-NCHW-implicit-prec: Size = 162, Time = 20.805 msec, Performance = 16721.6 GFlop/s
//cuDNN-NHWC-implicit-prec: Size = 162, Time = 20.245 msec, Performance = 17184.1 GFlop/s
//cuDNN-NHWC-GEMM-implicit: Size = 162, Time = 106.27 msec, Performance = 3273.7 GFlop/s

//[4] for: Feature = (60, 60), [N, IC, OC] = [128, 64, 64]
//LB = 4: Size = 142.383, Time = 11.1839 msec, Performace = 27339.8 GFlop/s
//WB = 4: Size = 142.383, Time = 10.3251 msec, Performace = 29613.7 GFlop/s
//cuDNN-NCHW-implicit-prec: Size = 142.382813, Time = 18.475 msec, Performance = 16550.2 GFlop/s
//cuDNN-NHWC-implicit-prec: Size = 142.382813, Time = 17.735 msec, Performance = 17240.8 GFlop/s
//cuDNN-NHWC-GEMM-implicit: Size = 142.382813, Time = 92.565 msec, Performance = 3303.2 GFlop/s

//[5] for: Feature = (48, 48), [N, IC, OC] = [128, 64, 64]
//LB = 4: Size = 91.125, Time = 6.25559 msec, Performace = 31282.3 GFlop/s
//WB = 4: Size = 91.125, Time = 5.86903 msec, Performace = 33342.7 GFlop/s
//cuDNN-NCHW-implicit-prec: Size = 91.125, Time = 11.98 msec, Performance = 16334.7 GFlop/s
//cuDNN-NHWC-implicit-prec: Size = 91.125, Time = 11.59 msec, Performance = 16884.3 GFlop/s
//cuDNN-NHWC-GEMM-implicit: Size = 91.125, Time = 59.015 msec, Performance = 3315.9 GFlop/s

//[6] for: Feature = (32, 32), [N, IC, OC] = [128, 128, 128]
//LB = 4: Size = 162, Time = 12.919 msec, Performace = 26928.6 GFlop/s
//WB = 4: Size = 162, Time = 11.977 msec, Performace = 29046.6 GFlop/s
//cuDNN-NCHW-implicit-prec: Size = 162, Time = 20.29 msec, Performance = 17146.0 GFlop/s
//cuDNN-NHWC-implicit-prec: Size = 162, Time = 19.6  msec, Performance = 17749.6 GFlop/s
//cuDNN-NHWC-GEMM-implicit: Size = 162, Time = 55.865 msec, Performance = 6227.4 GFlop/s

//[7] for: Feature = (28, 28), [N, IC, OC] = [128, 128, 128]
//LB = 4: Size = 124.03125, Time = 11.6123 msec, Performace = 22937.3 GFlop/s
//WB = 4: Size = 124.03125, Time = 10.9252 msec, Performace = 24379.8 GFlop/s
//cuDNN-NCHW-implicit-prec: Size = 124.03125, Time = 15.69  msec, Performance = 16976.1 GFlop/s
//cuDNN-NHWC-implicit-prec: Size = 124.03125, Time = 15.345 msec, Performance = 17357.8 GFlop/s
//cuDNN-NHWC-GEMM-implicit: Size = 124.03125, Time = 42.635 msec, Performance = 6247.3 GFlop/s

//[8] for: Feature = (16, 16), [N, IC, OC] = [128, 256, 256]
//LB = 4: Size = 162, Time = 12.557  msec, Performace = 27705   GFlop/s
//WB = 4: Size = 162, Time = 11.7536 msec, Performace = 29598.7 GFlop/s
//cuDNN-NCHW-implicit-prec: Size = 162, Time = 19.675 msec, Performance = 17681.9 GFlop/s
//cuDNN-NHWC-implicit-prec: Size = 162, Time = 20.29  msec, Performance = 17146.0 GFlop/s
//cuDNN-NHWC-GEMM-implicit: Size = 162, Time = 62.395 msec, Performance = 5575.6 GFlop/s

//[9] for: Feature = (8, 8), [N, IC, OC] = [128, 256, 256]
//LB = 4: Size = 162, Time = 11.6811 msec, Performace = 29782.5 GFlop/s
//WB = 4: Size = 162, Time = 10.6092 msec, Performace = 32791.5 GFlop/s
//cuDNN-NCHW-implicit-prec: Size = 162, Time = 19.675 msec, Performance = 17681.9 GFlop/s
//cuDNN-NHWC-implicit-prec: Size = 162, Time = 20.29  msec, Performance = 17146.0 GFlop/s
//cuDNN-NHWC-GEMM-implicit: Size = 162, Time = 62.395 msec, Performance = 5575.6 GFlop/s

#endif

//(1) FW = 9
//(2) sh = sw = 1
//(3) GN: (OC    ) % 64
//(4) GM: (N * OH) % 32
//(5) IC % 8 == 0
//(6) pw <= 4) && (OW - IW - pw + 4 <= 0)
//(7) OW >= 8
#ifndef COMPILE 
#define CONV_3D_WINOGRAD_S16_W9_C64X32R_P4_TEX
#endif
#ifndef CONV_3D_WINOGRAD_S16_W9_C64X32R_P4_TEX
#define CONV_3D_WINOGRAD_S16_W9_C64X32R_P4_TEX

template<int FH>
inline void conv3D_Winograd_s16_W9_64x32R_p4_tex(jlong* streams, int &index, int length,
	const float*  X, cudaTextureObject_t texX, int IH, int IW,
	const float* CW,//[FH, FW, IC, OC]
	      float*  Y, int OH, int OW,
	int N, int IC, int OC,
	int ph, int pw,
	int GN, int GM, int &GNr, int &GMr)
{
	//======[Stage1 (OW % 8) : F(8, 9) ]============================================================================
	next_cudaStream(stream, streams, index, length);
	conv3dWinograd_f8x9_k64x256R_p4_tex(stream, texX, IH, IW, CW, FH, Y, OH, OW, N, IC, OC, ph, pw);//common

	//======[Stage2 (OW % 4): Winograd_F(2+2, 3)]=======================================================================
	int OWr = OW & 7;
	if ((OWr >= 4) && !((N*OH) & 31)) {//Remainder: 4, 5, 6, 7, 8
		conv3dWinograd_SFW_f2x3_k64x128R4C_tex(stream, (OW - OWr), texX, IH, IW, CW, FH, 9, Y, OH, OW, N, IC, OC, ph, pw, OWr);
		OWr = OWr & 3;//OWr % 4
	}

	//======[Stage3 (OW % 2): Winograd_F(2, 3)]=========================================================================
	if ((OWr >= 2) && !((N*OH) & 63)) {//Remainder: 2, 3
		conv3dWinograd_SFW_f2x3_k64x128RC_tex(stream, (OW - OWr), texX, IH, IW, CW, FH, 9, Y, OH, OW, N, IC, OC, ph, pw, OWr);
		OWr = OWr & 1;//OWr % 2
	}

	//======[Stage4: GEMM]==============================================================================================
	if (OWr > 0) {//Remainder: 1
		const int GN64 = ((OC  ) >> 6 << 6);      //GN   = 64x
		const int GM32 = ((N*OH) >> 5 << 5) * OWr;//N*OH = 32x
		index = 0;//save L2 cache: texX != X
		conv3D_Gemm_32x32RC(streams, index, length,
			X, IH, IW, CW, FH, 9, Y, OH, OW, OWr, IC, OC, 1, 1, ph, pw,
			GN64, GM32, 0, 0, OW - OWr);
	}

	//=====[Process GNr and GMr]========================================================================================
	GNr = GN & 63;         //64 channels
	GMr = ((N*OH) & 31)*OW;//32 groups
}

#endif

#endif

#endif

#endif