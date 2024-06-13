#pragma once

#ifndef DECONV_3D_DELTAX_IM2COL_WINOGRAD_H
#define DECONV_3D_DELTAX_IM2COL_WINOGRAD_H

//process remainder of OW
#include "dconv3D_dX_ZeroPadding_uernel_s1_C.cuh"

//4 state
#include "dconv3D_dX_ori_Winograd_f2x3.cuh"//FW = 3
#include "dconv3D_dX_Im2col_Winograd_s4_f3x2.cuh"//FW = 2
#include "dconv3D_dX_Im2col_Winograd_s4_f2x3.cuh"//FW = 3

//8 state
#include "dconv3D_dX_Im2col_Winograd_s8_f7x2.cuh"//FW = 2
#include "dconv3D_dX_Im2col_Winograd_s8_f6x3.cuh"//FW = 3
#include "dconv3D_dX_Im2col_Winograd_s8_f5x4.cuh"//FW = 4
#include "dconv3D_dX_Im2col_Winograd_s8_f4x5.cuh"//FW = 5
#include "dconv3D_dX_Im2col_Winograd_s8_f3x6.cuh"//FW = 6
#include "dconv3D_dX_Im2col_Winograd_s8_f2x7.cuh"//FW = 7

//16 state
#include "dconv3D_dX_Im2col_Winograd_sg_fAx7.cuh"//FW = 7
#include "dconv3D_dX_Im2col_Winograd_sg_f9x8.cuh"//FW = 8
#include "dconv3D_dX_Im2col_Winograd_sg_f8x9.cuh"//FW = 9


//------[state = 8]-------------------------------------
#ifndef DECONV_3D_DX_WINOGRAD_S8_AREA
#define DECONV_3D_DX_WINOGRAD_S8_AREA

//FW = 2
#ifndef DECONV_3D_DX_WINOGRAD_S8_W2_64X32_TEXTURE
#define DECONV_3D_DX_WINOGRAD_S8_W2_64X32_TEXTURE

#ifndef DECONV_3D_DX_WINOGRAD_S8_W2_64X32_TEXTURE_BENCH_MARK
#define DECONV_3D_DX_WINOGRAD_S8_W2_64X32_TEXTURE_BENCH_MARK

//[0] for: Feature = (128, 128), [N, IC, OC] = [64, 64, 64]
//LB = 4: Size = 36, Time = 4.722 msec, Performace = 16372.2 GFlop/s

//[1] for: Feature = (64, 64), [N, IC, OC] = [256, 64, 64]
//LB = 4: Size = 32, Time = 5.577 msec, Performace = 12321.9 GFlop/s

//[2] for: Feature = (56, 56), [N, IC, OC] = [128, 128, 128]
//LB = 4: Size = 24.5, Time = 4.179 msec, Performace = 12589.9 GFlop/s

//[3] for: Feature = (32, 32), [N, IC, OC] = [256, 128, 128]
//LB = 4: Size = 32, Time = 5.108 msec, Performace = 13453.3 GFlop/s

//[4] for: Feature = (28, 28), [N, IC, OC] = [128, 256, 256]
//LB = 4: Size = 24.5, Time = 3.75 msec, Performace = 14030.2 GFlop/s

//[5] for: Feature = (16, 16), [N, IC, OC] = [256, 256, 256]
//LB = 4: Size = 32, Time = 4.928 msec, Performace = 13944.7 GFlop/s

//[6] for: Feature = (14, 14), [N, IC, OC] = [128, 512, 512]
//LB = 4: Size = 24.5, Time = 3.529 msec, Performace = 14908.9 GFlop/s

//[7] for: Feature = ( 8,  8), [N, IC, OC] = [256, 512, 512]
//LB = 4: Size = 32, Time = 4.847 msec, Performace = 14177.7 GFlop/s

#endif

//(1) FW = 2
//(2) sh = sw = 1
//(3) GN: (IC    ) % 64
//(4) GM: (N * IH) % 32
//(5) OC % 8 == 0
//(6) IW >= 7
#ifndef COMPILE
#define DECONV_3D_DX_WINOGRAD_S8_W2_64X32_TEX
#endif
#ifndef DECONV_3D_DX_WINOGRAD_S8_W2_64X32_TEX
#define DECONV_3D_DX_WINOGRAD_S8_W2_64X32_TEX

template<int FH>
inline void deconv3D_dX_winograd_s8_W2_64x32R_tex(jlong* streams, int &index, int length,
	const float*  deltaY, cudaTextureObject_t texDy, int OH, int OW,
	const float*       W,
	      float*  deltaX, int IH, int IW,
	int N, int IC, int OC,
	int ph, int pw,
	int GN, int GM, int &GMr, int &GNr)
{
	//======[Stage1 (IW % 7): Winograd_F(7, 2)]=========================================================================
	next_cudaStream(stream, streams, index, length);
	if ((pw >= 0) && (OW - IW - pw + 2 >= 0)) winograd_f7x2_k64x224_p0_tex(stream, texDy, OH, OW, W, FH, deltaX, IH, IW, N, IC, OC, ph, pw);
	else winograd_f7x2_k64x224_tex(stream, texDy, OH, OW, W, FH, deltaX, IH, IW, N, IC, OC, ph, pw);
	
	//======[Stage1 (IW % 6): Winograd_F(3 + 3, 2)]=====================================================================
	int IWr = IW % 7;
	if ((IWr >= 6) && !((N*IH) & 31)) {//Remainder: 4, 5, 6
		next_cudaStream(stream1, streams, index, length);
		winograd_f3x2_k64x192x6C_tex(stream1, IW - IWr, texDy, OH, OW, W, FH, deltaX, IH, IW, IC, OC, ph, pw, IWr);
		IWr = IWr % 6;//IWr % 6
	}

	//======[Stage2 (IW % 3): Winograd_F(3, 2)]=========================================================================
	if ((IWr >= 3) && !((N*IH) & 63)) {//Remainder: 2, 3
		next_cudaStream(stream2, streams, index, length);
		winograd_f3x2_k64x192C_tex(stream2, IW - IWr, texDy, OH, OW, W, FH, deltaX, IH, IW, IC, OC, ph, pw, IWr);
		IWr = IWr % 3;//IWr % 3
	}

	//======[Stage3: GEMM]==============================================================================================
	if (IWr > 0) {//Remainder: 1
		const int GN64 = ((IC) >> 6 << 6);        //GN   = 64x
		const int GM32 = ((N*IH) >> 5 << 5) * IWr;//N*IH = 32x
		index = 0;//save L2 cache: texDy != deltaY
		deconv3D_dX_ZeroPadding_s1_32x32C(streams, index, length,
			deltaY, OH, OW, W, FH, 2, deltaX, IH, IW, IWr, N, IC, OC, ph, pw,
			GN64, GM32, 0, 0, IW - IWr);
	}

	//=====[Process GNr and GMr]========================================================================================
	GNr = GN & 63;         //64 channels
	GMr = ((N*IH) & 31)*IW;//32 groups
}

#endif

#endif


//FW = 3
#ifndef DECONV_3D_DX_WINOGRAD_S8_W3_64X32_TEXTURE
#define DECONV_3D_DX_WINOGRAD_S8_W3_64X32_TEXTURE

#ifndef DECONV_3D_DX_WINOGRAD_S8_W3_64X32_TEXTURE_BENCK_MARK
#define DECONV_3D_DX_WINOGRAD_S8_W3_64X32_TEXTURE_BENCK_MARK

//[0] for: Feature = (128, 128), [N, IC, OC] = [64, 64, 64]
//LB = 4: Size = 36, Time = 4.722 msec, Performace = 16372.2 GFlop/s

//[1] for: Feature = (96, 96), [N, IC, OC] = [128, 64, 64]
//LB = 4: Size = 40.5, Time = 5.207 msec, Performace = 16703.1 GFlop/s

//[2] for: Feature = (64, 64), [N, IC, OC] = [256, 64, 164]
//LB = 4:Size = 36, Time = 4.795 msec, Performace = 16122.9 GFlop/s

//[3] for: Feature = (48, 48), [N, IC, OC] = [128, 128, 128]
//LB = 4: Size = 40.5, Time = 4.762 msec, Performace = 18264 GFlop/s

//[4] for: Feature = (32, 32), [N, IC, OC] = [256, 128, 128]
//LB = 4: Size = 36, Time = 4.389 msec, Performace = 17614.4 GFlop/s

//[5] for: Feature = (24, 24), [N, IC, OC] = [128, 256, 256]
//LB = 4: Size = 40.5, Time = 4.55 msec, Performace = 19115 GFlop/s

//[6] for: Feature = (16, 16), [N, IC, OC] = [256, 256, 256]
//LB = 4: Size = 36, Time = 4.34 msec, Performace = 17813.2 GFlop/s

//[7] for: Feature = ( 8,  8), [N, IC, OC] = [256, 512, 512]
//LB = 4: Size = 36, Time = 4.429 msec, Performace = 17455.3 GFlop/s

#endif

//(1) FW = 3
//(2) sh = sw = 1
//(3) GN: (IC    ) % 64
//(4) GM: (N * IH) % 32
//(5) OC % 8 == 0
//(6) IW >= 6
#ifndef COMPILE
#define DECONV_3D_DX_WINOGRAD_S8_W3_64X32_TEX
#endif
#ifndef DECONV_3D_DX_WINOGRAD_S8_W3_64X32_TEX
#define DECONV_3D_DX_WINOGRAD_S8_W3_64X32_TEX

template<int FH>
inline void deconv3D_dX_winograd_s8_W3_64x32R_tex(jlong* streams, int &index, int length,
	const float*  deltaY, cudaTextureObject_t texDy, int OH, int OW,
	const float*       W,
	      float*  deltaX, int IH, int IW,
	int N, int IC, int OC,
	int ph, int pw,
	int GN, int GM, int &GMr, int &GNr)
{
	//======[Stage1 (IW % 6): Winograd_F(6, 3)]=========================================================================
	next_cudaStream(stream, streams, index, length);
	if ((pw >= 1) && (OW - IW - pw + 1 >= 0)) {//pw >= 1
#ifdef ENABLE_DECONV3D_WINOGRAD_F6X3_CHANNEL_TEMPLATE
		//IC = 2^x
		if      (IC ==   64) winograd_f6x3_k64x192_p1_ICT_tex(stream, texDy, OH, OW, W, FH, deltaX, IH, IW, N,   64, OC, ph, pw);
		else if (IC ==  128) winograd_f6x3_k64x192_p1_ICT_tex(stream, texDy, OH, OW, W, FH, deltaX, IH, IW, N,  128, OC, ph, pw);
		else if (IC ==  256) winograd_f6x3_k64x192_p1_ICT_tex(stream, texDy, OH, OW, W, FH, deltaX, IH, IW, N,  256, OC, ph, pw);
		else if (IC ==  512) winograd_f6x3_k64x192_p1_ICT_tex(stream, texDy, OH, OW, W, FH, deltaX, IH, IW, N,  512, OC, ph, pw);
		else if (IC == 1024) winograd_f6x3_k64x192_p1_ICT_tex(stream, texDy, OH, OW, W, FH, deltaX, IH, IW, N, 1024, OC, ph, pw);
		//IC = 64x
		else if (IC ==  192) winograd_f6x3_k64x192_p1_ICT_tex(stream, texDy, OH, OW, W, FH, deltaX, IH, IW, N,  192, OC, ph, pw);
		else if (IC ==  320) winograd_f6x3_k64x192_p1_ICT_tex(stream, texDy, OH, OW, W, FH, deltaX, IH, IW, N,  320, OC, ph, pw);
		else if (IC ==  384) winograd_f6x3_k64x192_p1_ICT_tex(stream, texDy, OH, OW, W, FH, deltaX, IH, IW, N,  384, OC, ph, pw);
		else if (IC ==  448) winograd_f6x3_k64x192_p1_ICT_tex(stream, texDy, OH, OW, W, FH, deltaX, IH, IW, N,  448, OC, ph, pw);
		else
#endif
		winograd_f6x3_k64x192_p1_tex(stream, texDy, OH, OW, W, FH, deltaX, IH, IW, N, IC, OC, ph, pw);//common
	}
	else winograd_f6x3_k64x192_tex(stream, texDy, OH, OW, W, FH, deltaX, IH, IW, N, IC, OC, ph, pw);

	//======[Stage2 (IW % 2): Winograd_F(2 + 2, 3)]=====================================================================
	int IWr = IW % 6;
	if ((IWr >= 4) && !((N*IH) & 31)) {//Remainder: 3, 4, 5
		next_cudaStream(stream1, streams, index, length);
		winograd_f2x3_k64x128x4C_tex(stream1, IW - IWr, texDy, OH, OW, W, FH, deltaX, IH, IW, IC, OC, ph, pw, IWr);
		IWr = IWr & 3;//IWr % 4
	}

	//======[Stage3 (IW % 2): Winograd_F(2, 3)]=========================================================================
	if ((IWr >= 2) && !((N*IH) & 63)) {//Remainder: 2
		next_cudaStream(stream2, streams, index, length);
		winograd_f2x3_k64x128C_tex(stream2, IW - IWr, texDy, OH, OW, W, FH, deltaX, IH, IW, IC, OC, ph, pw, IWr);
		IWr = IWr & 1;//IWr % 2
	}

	//======[Stage4: GEMM]==============================================================================================
	if (IWr > 0) {//Remainder: 1
		const int GN64 = ((IC  ) >> 6 << 6);      //GN   = 64x
		const int GM32 = ((N*IH) >> 5 << 5) * IWr;//N*IH = 32x
		index = 0;//save L2 cache: texDy != deltaY
		deconv3D_dX_ZeroPadding_s1_32x32C(streams, index, length,
			deltaY, OH, OW, W, FH, 3, deltaX, IH, IW, IWr, N, IC, OC, ph, pw,
			GN64, GM32, 0, 0, IW - IWr);
	}

	//=====[Process GNr and GMr]========================================================================================
	GNr = GN & 63;         //64 channels
	GMr = ((N*IH) & 31)*IW;//32 groups
}

#endif

#endif


//FW = 4
#ifndef DECONV_3D_DX_WINOGRAD_S8_W4_64X32_TEXTURE
#define DECONV_3D_DX_WINOGRAD_S8_W4_64X32_TEXTURE

#ifndef DECONV_3D_DX_WINOGRAD_S8_W4_64X32_TEXTURE_BENCH_MARK
#define DECONV_3D_DX_WINOGRAD_S8_W4_64X32_TEXTURE_BENCH_MARK

//[0] for: Feature = (80, 80), [N, IC, OC] = [128, 64, 64]
//LB = 4: Size = 50, Time = 5.66  msec, Performace = 18970.7 GFlop/s

//[1] for: Feature = (64, 64), [N, IC, OC] = [128, 64, 64]
//LB = 4: Size = 32, Time = 3.771 msec, Performace = 18223.1 GFlop/s
//cuDNN-NHWC-implicit-prec: Size = 32, Time = 4.325 msec, Performance = 15888.9 GFlop/s

//[2] for: Feature = (40, 40), [N, IC, OC] = [128, 128, 128]
//LB = 4: Size = 50, Time = 5.293 msec, Performace = 20286.1 GFlop/s

//[3] for: Feature = (32, 32), [N, IC, OC] = [128, 128, 128]
//LB = 4: Size = 32, Time = 3.503 msec, Performace = 19617.3 GFlop/s
//cuDNN-NHWC-implicit-prec: Size = 32, Time = 4.529 msec, Performance = 15173.2 GFlop/s

//[4] for: Feature = (20, 20), [N, IC, OC] = [128, 256, 256]
//LB = 4: Size = 50, Time = 5.015 msec, Performace = 21410.6 GFlop/s

//[5] for: Feature = (16, 16), [N, IC, OC] = [128, 256, 256]
//LB = 4: Size = 32, Time = 3.674 msec, Performace = 18704.3 GFlop/s
//cuDNN-NHWC-implicit-prec: Size = 32, Time = 4.261 msec, Performance = 16127.5 GFlop/s

//[6] for: Feature = (8, 8), [N, IC, OC] = [128, 512, 512]
//LB = 4: Size = 32, Time = 4.045 msec, Performace = 16988.7 GFlop/s
//cuDNN-NHWC-implicit-prec: Size = 32, Time = 4.138 msec, Performance = 16606.9 GFlop/s

#endif

//(1) FW = 4
//(2) sh = sw = 1
//(3) GN: (IC    ) % 64
//(4) GM: (N * IH) % 32
//(5) OC % 8 == 0
//(6) IW >= 5
#ifndef COMPILE
#define DECONV_3D_DX_WINOGRAD_S8_W4_64X32_TEX
#endif
#ifndef DECONV_3D_DX_WINOGRAD_S8_W4_64X32_TEX
#define DECONV_3D_DX_WINOGRAD_S8_W4_64X32_TEX

template<int FH>
inline void deconv3D_dX_winograd_s8_W4_64x32R_tex(jlong* streams, int &index, int length,
	const float*  deltaY, cudaTextureObject_t texDy, int OH, int OW,
	const float*       W,
	      float*  deltaX, int IH, int IW,
	int N, int IC, int OC,
	int ph, int pw,
	int GN, int GM, int &GMr, int &GNr)
{
	//======[Stage1 (IW % 5): Winograd_F(5, 4)]=========================================================================
	next_cudaStream(stream, streams, index, length);
	if ((pw >= 1) && (OW - IW - pw + 3 >= 0)) winograd_f5x4_k64x160_p1_tex(stream, texDy, OH, OW, W, FH, deltaX, IH, IW, N, IC, OC, ph, pw);
	else winograd_f5x4_k64x160_tex(stream, texDy, OH, OW, W, FH, deltaX, IH, IW, N, IC, OC, ph, pw);

	//======[Stage2 (OW % 3): Winograd_F(3, 2)]=========================================================================
	int IWr = IW % 5;
	if ((IWr >= 3) && !((N*IH) & 63)) {//Remainder: 3, 4
		winograd_SFW_f3x2_k64x192C_tex(stream, (IW - IWr), texDy, OH, OW, W, FH, 4, deltaX, IH, IW, IC, OC, ph, pw, IWr);
		IWr = IWr % 3;//OWr % 3
	}

	//======[Stage3: GEMM]==============================================================================================
	if (IWr > 0) {//Remainder: 1, 2
		const int GN64 = ((IC  ) >> 6 << 6);      //GN   = 64x
		const int GM32 = ((N*IH) >> 5 << 5) * IWr;//N*IH = 32x
		index = 0;//save L2 cache: texDy != deltaY
		deconv3D_dX_ZeroPadding_s1_32x32C(streams, index, length,
			deltaY, OH, OW, W, FH, 4, deltaX, IH, IW, IWr, N, IC, OC, ph, pw,
			GN64, GM32, 0, 0, (IW - IWr));
	}

	//=====[Process GNr and GMr]========================================================================================
	GNr = GN & 63;         //64 channels
	GMr = ((N*IH) & 31)*IW;//32 groups
}

#endif

#endif


//FW = 5
#ifndef DECONV_3D_DX_WINOGRAD_S8_W5_64X32_TEXTURE
#define DECONV_3D_DX_WINOGRAD_S8_W5_64X32_TEXTURE

#ifndef DECONV_3D_DX_WINOGRAD_S8_W5_64X32_TEXTURE_BENCH_MARK
#define DECONV_3D_DX_WINOGRAD_S8_W5_64X32_TEXTURE_BENCH_MARK

//[0] for: Feature = (128, 128), [N, IC, OC] = [32, 64, 64]
//ruse = 4: Size = 50, Time = 4.674 msec, Performace = 22972.7 GFlop/s
//LB   = 4: Size = 50, Time = 5.373 msec, Performace = 19984 G  Flop/s

//[1] for: Feature = (66, 66), [N, IC, OC] = [128, 64, 64]
//ruse = 4: Size = 53.173828, Time = 5.494 msec, Performace = 20784.5 GFlop/s
//LB   = 4: Size = 53.173828, Time = 6.445 msec, Performace = 17717.6 GFlop/s

//[2] for: Feature = (64, 64), [N, IC, OC] = [32, 128, 128]
//LB   = 4: Size = 50, Time = 5.4   msec, Performace = 19884.1 GFlop/s
//ruse = 4: Size = 50, Time = 4.645 msec, Performace = 23116.1 GFlop/s

//[3] for: Feature = (34, 34), [N, IC, OC] = [128, 128, 128]
//ruse = 4: Size = 56.445313, Time = 6.081 msec, Performace = 19933.5 GFlop/s
//LB   = 4: Size = 56.445313, Time = 6.379 msec, Performace = 19002.3 GFlop/s

//[4] for: Feature = (32, 32), [N, IC, OC] = [128, 128, 128]
//LB   = 4: Size = 50, Time = 5.214 msec, Performace = 20593.4 GFlop/s
//ruse = 4: Size = 50, Time = 4.827 msec, Performace = 22244.5 GFlop/s

//[5] for: Feature = (18, 18), [N, IC, OC] = [128, 256, 256]
//ruse = 4: Size = 63.28125, Time = 6.604 msec, Performace = 20577.7 GFlop/s
//LB   = 4: Size = 63.28125, Time = 7.281 msec, Performace = 18664.4 GFlop/s

//[6] for: Feature = (16, 16), [N, IC, OC] = [128, 256, 256]
//ruse = 4: Size = 50, Time = 4.88 msec, Performace = 22002.9 GFlop/s

//for: Feature = (10, 10), [N, IC, OC] = [128, 512, 512]
//ruse = 4: Size = 78.125, Time =  9.327 msec, Performace = 17987.8 GFlop/s
//LB   = 4: Size = 78.125, Time = 10.248 msec, Performace = 16371.2 GFlop/s

//for: Feature = ( 8,  8), [N, IC, OC] = [128, 512, 512]
//ruse = 4: Size = 50, Time = 4.518 msec, Performace = 23765.9 GFlop/s
//LB   = 4: Size = 50, Time = 5.13  msec, Performace = 20930.6 GFlop/s

#endif

//(1) FW = 5
//(2) sh = sw = 1
//(3) GN: (IC    ) % 64
//(4) GM: (N * IH) % 32
//(5) OC % 8 == 0
//(6) IW >= 4
#ifndef COMPILE
#define DECONV_3D_DX_WINOGRAD_S8_W5_64X32_TEX
#endif
#ifndef DECONV_3D_DX_WINOGRAD_S8_W5_64X32_TEX
#define DECONV_3D_DX_WINOGRAD_S8_W5_64X32_TEX

template<int FH>
inline void deconv3D_dX_winograd_s8_W5_64x32R_tex(jlong* streams, int &index, int length,
	const float*  deltaY, cudaTextureObject_t texDy, int OH, int OW,
	const float*       W,
	      float*  deltaX, int IH, int IW,
	int N, int IC, int OC,
	int ph, int pw,
	int GN, int GM, int &GMr, int &GNr) 
{
	//======[Stage1 (IW % 4): Winograd_F(4, 5)]=========================================================================
	next_cudaStream(stream, streams, index, length);
	if ((pw >= 2) && (OW - IW - pw + 2 >= 0)) {//pw >= 2
		if (IW >= 8) {
			winograd_f4x5_ruse_k64x128_p2_tex(stream, texDy, OH, OW, W, FH, deltaX, IH, IW, N, IC, OC, ph, pw);
			const int IWr = IW & 7;//align: IW % 4
			if (IWr >= 4) winograd_f4x5_k64x128C_p2_tex(stream, (IW - IWr), texDy, OH, OW, W, FH, deltaX, IH, IW, N, IC, OC, ph, pw, IWr);
		}
		else winograd_f4x5_k64x128_p2_tex(stream, texDy, OH, OW, W, FH, deltaX, IH, IW, N, IC, OC, ph, pw);
	}
	else winograd_f4x5_k64x128_tex(stream, texDy, OH, OW, W, FH, deltaX, IH, IW, N, IC, OC, ph, pw);

	//======[Stage2: GEMM]==============================================================================================
	const int IWr = IW & 3;
	if (IWr > 0) {//Remainder: 1, 2, 3
		const int GN64 = ((IC  ) >> 6 << 6);      //GN   = 64x
		const int GM32 = ((N*IH) >> 5 << 5) * IWr;//N*IH = 32x
		index = 0;//save L2 cache: texDy != deltaY
		deconv3D_dX_ZeroPadding_s1_32x32C(streams, index, length,
			deltaY, OH, OW, W, FH, 5, deltaX, IH, IW, IWr, N, IC, OC, ph, pw,
			GN64, GM32, 0, 0, IW - IWr);
	}

	//=====[Process GNr and GMr]========================================================================================
	GNr = GN & 63;         //64 channels
	GMr = ((N*IH) & 31)*IW;//32 groups
}

#endif

#endif


//FW = 6
#ifndef DECONV_3D_DX_WINOGRAD_S8_W6_64X32_TEXTURE
#define DECONV_3D_DX_WINOGRAD_S8_W6_64X32_TEXTURE

#ifndef DECONV_3D_DX_WINOGRAD_S8_W6_64X32_TEXTURE_BENCH_MARK
#define DECONV_3D_DX_WINOGRAD_S8_W6_64X32_TEXTURE_BENCH_MARK

//[0] for: Feature = (64, 64), [N, IC, OC] = [128, 64, 64]
//ruse = 4: Size = 72, Time = 7.385 msec, Performace = 20936.9 GFlop/s

//[1] for: Feature = (48, 48), [N, IC, OC] = [128, 64, 64]
//ruse = 4: Size = 40.5, Time = 4.105 msec, Performace = 21187.1 GFlop/s
//LB   = 4: Size = 40.5, Time = 4.105 msec, Performace = 21187.1 GFlop/s

//[2] for: Feature = (32, 32), [N, IC, OC] = [128, 128, 128]
//ruse = 4: Size = 72, Time = 7.674 msec, Performace = 20148.4 GFlop/s

//[3] for: Feature = (24, 24), [N, IC, OC] = [128, 128, 128]
//ruse = 4: Size = 40.5, Time = 4.175 msec, Performace = 20831.9 GFlop/s
//LB   = 4: Size = 40.5, Time = 4.175 msec, Performace = 20831.9 GFlop/s

//[4] for: Feature = (16, 16), [N, IC, OC] = [128, 256, 256]
//ruse = 4: Size = 72, Time = 7.232 msec, Performace = 21379.8 GFlop/s

//[5] for: Feature = (12, 12), [N, IC, OC] = [128, 256, 256]
//ruse = 4: Size = 40.5, Time = 4.252 msec, Performace = 20454.6 GFlop/s
//LB   = 4: Size = 40.5, Time = 4.252 msec, Performace = 20454.6 GFlop/s

//[6] for: Feature = (8, 8), [N, IC, OC] = [128, 512, 512]
//ruse = 4: Size = 72, Time = 8.27 msec, Performace = 18696.3 GFlop/s

#endif

//(1) FW = 6
//(2) sh = sw = 1
//(3) GN: (IC    ) % 64
//(4) GM: (N * IH) % 32
//(5) OC % 8 == 0
//(6) IW >= 3
#ifndef COMPILE
#define DECONV_3D_DX_WINOGRAD_S8_W6_64X32_TEX
#endif
#ifndef DECONV_3D_DX_WINOGRAD_S8_W6_64X32_TEX
#define DECONV_3D_DX_WINOGRAD_S8_W6_64X32_TEX

template<int FH>
inline void deconv3D_dX_winograd_s8_W6_64x32R_tex(jlong* streams, int &index, int length,
	const float*  deltaY, cudaTextureObject_t texDy, int OH, int OW,
	const float*       W,
	      float*  deltaX, int IH, int IW,
	int N, int IC, int OC,
	int ph, int pw,
	int GN, int GM, int &GMr, int &GNr)
{
	//======[Stage1 (OW % 3): Winograd_F(3, 6)]=========================================================================
	next_cudaStream(stream, streams, index, length);
	if ((pw >= 2) && (OW - IW - pw + 4 >= 0) && (OW >= 6)) {//pw >= 2
		winograd_f3x6_ruse_k64x96_p2_tex(stream, texDy, OH, OW, W, FH, deltaX, IH, IW, N, IC, OC, ph, pw);
		const int IWr = IW % 6;//align: IW % 3
		if (IWr >= 3) { int iw_index = (IW - IWr);
#ifdef ENABLE_DECONV3D_WINOGRAD_F3X6_CHANNEL_TEMPLATE
			//IC = 2^x
			if      (IC ==  64 && OC ==  64) winograd_f3x6_k64x96C_CT_tex(stream, iw_index, texDy, OH, OW, W, FH, deltaX, IH, IW, N,  64,  64, ph, pw, IWr);
			else if (IC == 128 && OC == 128) winograd_f3x6_k64x96C_CT_tex(stream, iw_index, texDy, OH, OW, W, FH, deltaX, IH, IW, N, 128, 128, ph, pw, IWr);
			else if (IC == 256 && OC == 256) winograd_f3x6_k64x96C_CT_tex(stream, iw_index, texDy, OH, OW, W, FH, deltaX, IH, IW, N, 256, 256, ph, pw, IWr);
			else if (IC == 512 && OC == 512) winograd_f3x6_k64x96C_CT_tex(stream, iw_index, texDy, OH, OW, W, FH, deltaX, IH, IW, N, 512, 512, ph, pw, IWr);
			//IC = 64x
			else if (IC == 192 && OC == 192) winograd_f3x6_k64x96C_CT_tex(stream, iw_index, texDy, OH, OW, W, FH, deltaX, IH, IW, N, 192, 192, ph, pw, IWr);
			else if (IC == 320 && OC == 320) winograd_f3x6_k64x96C_CT_tex(stream, iw_index, texDy, OH, OW, W, FH, deltaX, IH, IW, N, 320, 320, ph, pw, IWr);
			else if (IC == 384 && OC == 384) winograd_f3x6_k64x96C_CT_tex(stream, iw_index, texDy, OH, OW, W, FH, deltaX, IH, IW, N, 384, 384, ph, pw, IWr);
			else if (IC == 448 && OC == 448) winograd_f3x6_k64x96C_CT_tex(stream, iw_index, texDy, OH, OW, W, FH, deltaX, IH, IW, N, 448, 448, ph, pw, IWr);
			else
#endif
			winograd_f3x6_k64x96C_tex(stream, iw_index, texDy, OH, OW, W, FH, deltaX, IH, IW, N, IC, OC, ph, pw, IWr);
		}
	}
#ifdef ENABLE_DECONV3D_WINOGRAD_F3X6_CHANNEL_TEMPLATE
	//IC = 2^x
	else if (IC ==  64) { if (OC ==  64) winograd_f3x6_k64x96_CT_tex(stream, texDy, OH, OW, W, FH, deltaX, IH, IW, N,  64,  64, ph, pw); else winograd_f3x6_k64x96_ICT_tex(stream, texDy, OH, OW, W, FH, deltaX, IH, IW, N, 64, OC, ph, pw); }
	else if (IC == 128) { if (OC == 128) winograd_f3x6_k64x96_CT_tex(stream, texDy, OH, OW, W, FH, deltaX, IH, IW, N, 128, 128, ph, pw); else winograd_f3x6_k64x96_ICT_tex(stream, texDy, OH, OW, W, FH, deltaX, IH, IW, N, 128, OC, ph, pw); }
	else if (IC == 256) { if (OC == 256) winograd_f3x6_k64x96_CT_tex(stream, texDy, OH, OW, W, FH, deltaX, IH, IW, N, 256, 256, ph, pw); else winograd_f3x6_k64x96_ICT_tex(stream, texDy, OH, OW, W, FH, deltaX, IH, IW, N, 256, OC, ph, pw); }
	else if (IC == 512) { if (OC == 512) winograd_f3x6_k64x96_CT_tex(stream, texDy, OH, OW, W, FH, deltaX, IH, IW, N, 512, 512, ph, pw); else winograd_f3x6_k64x96_ICT_tex(stream, texDy, OH, OW, W, FH, deltaX, IH, IW, N, 512, OC, ph, pw); }
	//IC = 64x
	else if (IC == 192) { if (OC == 192) winograd_f3x6_k64x96_CT_tex(stream, texDy, OH, OW, W, FH, deltaX, IH, IW, N, 192, 192, ph, pw); else winograd_f3x6_k64x96_ICT_tex(stream, texDy, OH, OW, W, FH, deltaX, IH, IW, N, 192, OC, ph, pw); }
	else if (IC == 320) { if (OC == 320) winograd_f3x6_k64x96_CT_tex(stream, texDy, OH, OW, W, FH, deltaX, IH, IW, N, 320, 320, ph, pw); else winograd_f3x6_k64x96_ICT_tex(stream, texDy, OH, OW, W, FH, deltaX, IH, IW, N, 320, OC, ph, pw); }
	else if (IC == 384) { if (OC == 384) winograd_f3x6_k64x96_CT_tex(stream, texDy, OH, OW, W, FH, deltaX, IH, IW, N, 384, 384, ph, pw); else winograd_f3x6_k64x96_ICT_tex(stream, texDy, OH, OW, W, FH, deltaX, IH, IW, N, 384, OC, ph, pw); }
	else if (IC == 448) { if (OC == 448) winograd_f3x6_k64x96_CT_tex(stream, texDy, OH, OW, W, FH, deltaX, IH, IW, N, 448, 448, ph, pw); else winograd_f3x6_k64x96_ICT_tex(stream, texDy, OH, OW, W, FH, deltaX, IH, IW, N, 448, OC, ph, pw); }
#endif
	else winograd_f3x6_k64x96_tex(stream, texDy, OH, OW, W, FH, deltaX, IH, IW, N, IC, OC, ph, pw);//common

	//======[Stage2 (OW % 2): Winograd_F(2, 3)]=========================================================================
	int IWr = IW % 3;
	if ((IWr >= 2) && !((N*IH) & 63)) {//Remainder: 2
		winograd_SFW_f2x3_k64x128C_tex(stream, (IW - IWr), texDy, OH, OW, W, FH, 6, deltaX, IH, IW, IC, OC, ph, pw, IWr);
		IWr = IWr & 1;//OWr % 2
	}

	//======[Stage3: GEMM]==============================================================================================
	if (IWr > 0) {//Remainder: 1
		const int GN64 = ((IC  ) >> 6 << 6);      //GN   = 64x
		const int GM32 = ((N*IH) >> 5 << 5) * IWr;//N*IH = 32x
		index = 0;//save L2 cache: texDy != deltaY
		deconv3D_dX_ZeroPadding_s1_32x32C(streams, index, length,
			deltaY, OH, OW, W, FH, 6, deltaX, IH, IW, IWr, N, IC, OC, ph, pw,
			GN64, GM32, 0, 0, IW - IWr);
	}

	//=====[Process GNr and GMr]========================================================================================
	GNr = GN & 63;         //64 channels
	GMr = ((N*IH) & 31)*IW;//32 groups
}

#endif

#endif


//FW = 7
#ifndef DECONV_3D_DX_WINOGRAD_S8_W7_64X32_TEXTURE
#define DECONV_3D_DX_WINOGRAD_S8_W7_64X32_TEXTURE

#ifndef DECONV_3D_DX_WINOGRAD_S8_W7_64X32_TEXTURE_BENCH_MARK
#define DECONV_3D_DX_WINOGRAD_S8_W7_64X32_TEXTURE_BENCH_MARK

//for: feature = (34, 34), [N, IC, OC] = [64, 128, 128]
//LB = 4: Size = 55.3164, Time =  7.207 msec, Performace = 16482.7 GFlop/s

//for: feature = (18, 18), [N, IC, OC] = [64, 256, 256]
//LB = 4: Size = 62.0156, Time =  8.162 msec, Performace = 16316.8 GFlop/s

//for: feature = (10, 10), [N, IC, OC] = [64, 512, 512]
//LB = 4: Size = 76.5625, Time = 10.374 msec, Performace = 15848.9 GFlop/s

#endif

//(1) FW = 7
//(2) sh = sw = 1
//(3) GN: (IC    ) % 64
//(4) GM: (N * IH) % 32
//(5) OC % 8 == 0
//(6) IW >= 2
#ifndef COMPILE
#define DECONV_3D_DX_WINOGRAD_S8_W7_64X32_TEX
#endif
#ifndef DECONV_3D_DX_WINOGRAD_S8_W7_64X32_TEX
#define DECONV_3D_DX_WINOGRAD_S8_W7_64X32_TEX

template<int FH>
inline void deconv3D_dX_winograd_s8_W7_64x32R_tex(jlong* streams, int &index, int length,
	const float*  deltaY, cudaTextureObject_t texDy, int OH, int OW,
	const float*       W,
	      float*  deltaX, int IH, int IW,
	int N, int IC, int OC,
	int ph, int pw,
	int GN, int GM, int &GMr, int &GNr)
{
	//======[Stage1 (IW % 2): Winograd_F(2, 7)]=========================================================================
	next_cudaStream(stream, streams, index, length);
	if ((pw >= 3) && (OW - IW - pw + 3 >= 0) && (IW >= 4)) {//pw >= 3
#ifdef ENABLE_DECONV3D_WINOGRAD_F2X7_CHANNEL_TEMPLATE
		if      (IC ==  64 && OC ==  64) winograd_f2x7_ruse_k64x64_p3_CT_tex(stream, texDy, OH, OW, W, FH, deltaX, IH, IW, N,  64,  64, ph, pw);
		else if (IC == 128 && OC == 128) winograd_f2x7_ruse_k64x64_p3_CT_tex(stream, texDy, OH, OW, W, FH, deltaX, IH, IW, N, 128, 128, ph, pw);
		else if (IC == 256 && OC == 256) winograd_f2x7_ruse_k64x64_p3_CT_tex(stream, texDy, OH, OW, W, FH, deltaX, IH, IW, N, 256, 256, ph, pw);
		else if (IC == 512 && OC == 512) winograd_f2x7_ruse_k64x64_p3_CT_tex(stream, texDy, OH, OW, W, FH, deltaX, IH, IW, N, 512, 512, ph, pw);
		//IC = 64x
		else if (IC == 192 && OC == 192) winograd_f2x7_ruse_k64x64_p3_CT_tex(stream, texDy, OH, OW, W, FH, deltaX, IH, IW, N, 192, 192, ph, pw);
		else if (IC == 320 && OC == 320) winograd_f2x7_ruse_k64x64_p3_CT_tex(stream, texDy, OH, OW, W, FH, deltaX, IH, IW, N, 320, 320, ph, pw);
		else if (IC == 384 && OC == 384) winograd_f2x7_ruse_k64x64_p3_CT_tex(stream, texDy, OH, OW, W, FH, deltaX, IH, IW, N, 384, 384, ph, pw);
		else if (IC == 448 && OC == 448) winograd_f2x7_ruse_k64x64_p3_CT_tex(stream, texDy, OH, OW, W, FH, deltaX, IH, IW, N, 448, 448, ph, pw);
		else
#endif
		winograd_f2x7_ruse_k64x64_p3_tex(stream, texDy, OH, OW, W, FH, deltaX, IH, IW, N, IC, OC, ph, pw);//common

		const int IWr = IW & 3;//align: IW % 2
		if (IWr >= 2) { const int iw_index = IW - IWr;
#ifdef ENABLE_DECONV3D_WINOGRAD_F2X7_CHANNEL_TEMPLATE
			//IC = 2^x
			if      (IC ==  64 && OC ==  64) winograd_f2x7_k64x64C_CT_tex(stream, iw_index, texDy, OH, OW, W, FH, deltaX, IH, IW, N,  64,  64, ph, pw, IWr);
			else if (IC == 128 && OC == 128) winograd_f2x7_k64x64C_CT_tex(stream, iw_index, texDy, OH, OW, W, FH, deltaX, IH, IW, N, 128, 128, ph, pw, IWr);
			else if (IC == 256 && OC == 256) winograd_f2x7_k64x64C_CT_tex(stream, iw_index, texDy, OH, OW, W, FH, deltaX, IH, IW, N, 256, 256, ph, pw, IWr);
			else if (IC == 512 && OC == 512) winograd_f2x7_k64x64C_CT_tex(stream, iw_index, texDy, OH, OW, W, FH, deltaX, IH, IW, N, 512, 512, ph, pw, IWr);
			//IC = 64x
			else if (IC == 192 && OC == 192) winograd_f2x7_k64x64C_CT_tex(stream, iw_index, texDy, OH, OW, W, FH, deltaX, IH, IW, N, 192, 192, ph, pw, IWr);
			else if (IC == 320 && OC == 320) winograd_f2x7_k64x64C_CT_tex(stream, iw_index, texDy, OH, OW, W, FH, deltaX, IH, IW, N, 320, 320, ph, pw, IWr);
			else if (IC == 384 && OC == 384) winograd_f2x7_k64x64C_CT_tex(stream, iw_index, texDy, OH, OW, W, FH, deltaX, IH, IW, N, 384, 384, ph, pw, IWr);
			else if (IC == 448 && OC == 448) winograd_f2x7_k64x64C_CT_tex(stream, iw_index, texDy, OH, OW, W, FH, deltaX, IH, IW, N, 448, 448, ph, pw, IWr);
			else
#endif
			winograd_f2x7_k64x64C_tex(stream, iw_index, texDy, OH, OW, W, FH, deltaX, IH, IW, N, IC, OC, ph, pw, IWr);//common
		}
	}
#ifdef ENABLE_DECONV3D_WINOGRAD_F2X7_CHANNEL_TEMPLATE
	//IC = 2^x
	else if      (IC ==  64 && OC ==  64) winograd_f2x7_k64x64_CT_tex(stream, texDy, OH, OW, W, FH, deltaX, IH, IW, N,  64,  64, ph, pw);
	else if (IC == 128 && OC == 128) winograd_f2x7_k64x64_CT_tex(stream, texDy, OH, OW, W, FH, deltaX, IH, IW, N, 128, 128, ph, pw);
	else if (IC == 256 && OC == 256) winograd_f2x7_k64x64_CT_tex(stream, texDy, OH, OW, W, FH, deltaX, IH, IW, N, 256, 256, ph, pw);
	else if (IC == 512 && OC == 512) winograd_f2x7_k64x64_CT_tex(stream, texDy, OH, OW, W, FH, deltaX, IH, IW, N, 512, 512, ph, pw);
	//IC = 64x
	else if (IC == 192 && OC == 192) winograd_f2x7_k64x64_CT_tex(stream, texDy, OH, OW, W, FH, deltaX, IH, IW, N, 192, 192, ph, pw);
	else if (IC == 320 && OC == 320) winograd_f2x7_k64x64_CT_tex(stream, texDy, OH, OW, W, FH, deltaX, IH, IW, N, 320, 320, ph, pw);
	else if (IC == 384 && OC == 384) winograd_f2x7_k64x64_CT_tex(stream, texDy, OH, OW, W, FH, deltaX, IH, IW, N, 384, 384, ph, pw);
	else if (IC == 448 && OC == 448) winograd_f2x7_k64x64_CT_tex(stream, texDy, OH, OW, W, FH, deltaX, IH, IW, N, 448, 448, ph, pw);
#endif
	else winograd_f2x7_k64x64_tex(stream, texDy, OH, OW, W, FH, deltaX, IH, IW, N, IC, OC, ph, pw);//common

	//======[Stage2: GEMM]================================================================================================
	const int IWr = IW & 1;
	if (IWr > 0) {//Remainder: 1
		const int GN64 = ((IC  ) >> 6 << 6);      //GN   = 64x
		const int GM32 = ((N*IH) >> 5 << 5) * IWr;//N*IH = 32x
		index = 0;//save L2 cache: texDy != deltaY
		index = 0;//save L2 cache: texDy != deltaY
		deconv3D_dX_ZeroPadding_s1_32x32C(streams, index, length,
			deltaY, OH, OW, W, FH, 7, deltaX, IH, IW, IWr, N, IC, OC, ph, pw,
			GN64, GM32, 0, 0, IW - IWr);
	}

	//=====[Process GNr and GMr]========================================================================================
	GNr = GN & 63;         //64 channels
	GMr = ((N*IH) & 31)*IW;//32 groups
}

#endif

#endif

#endif


//------[state = 16]------------------------------------
#ifndef DECONV_3D_DX_WINOGRAD_S16_AREA
#define DECONV_3D_DX_WINOGRAD_S16_AREA

//FW == 7
#ifndef DECONV_3D_WINOGRAD_S16_W7_32X32_P3_TEXTURE
#define DECONV_3D_WINOGRAD_S16_W7_32X32_P3_TEXTURE

#ifndef DECONV_3D_WINOGRAD_S16_W7_32X32_P3_TEXTURE_BENCH_MARK
#define DECONV_3D_WINOGRAD_S16_W7_32X32_P3_TEXTURE_BENCH_MARK

//[0] for: Feature = (128, 128), [N, IC, OC] = [32, 64, 64]
//LB = 4: Size = 49, Time = 4.247 msec, Performace = 24776.7 GFlop/s

//[1] for: Feature = (120, 120), [N, IC, OC] = [32, 64, 64]
//LB = 4: Size = 86.1328, Time = 7.17 msec, Performace = 25797.6 GFlop/s

//[2] for: Feature = (80, 80), [N, IC, OC] = [64, 64, 64]
//LB = 4: Size = 76.5625, Time = 6.185 msec, Performace = 26583.1 GFlop/s

//[3] for: Feature = (64, 64), [N, IC, OC] = [64, 64, 64]
//LB = 4: Size = 49, Time = 4.084 msec, Performace = 25765.6 GFlop/s

//[4] for: Feature = (40, 40), [N, IC, OC] = [64, 128, 128]
//LB = 4: Size = 76.5625, Time = 7.275 msec, Performace = 22600.2 GFlop/s

//[5] for: Feature = (32, 32), [N, IC, OC] = [64, 128, 128]
//LB = 4: Size = 98, Time = 9.781 msec, Performace = 21516.6 GFlop/s

//[6] for: Feature = (20, 20), [N, IC, OC] = [64, 256, 256]
//LB = 4: Size = 76.5625, Time = 7.216 msec, Performace = 22785 GFlop/s

//[7] for: Feature = (16, 16), [N, IC, OC] = [64, 256, 256]
//LB = 4: Size = 49, Time = 5.535 msec, Performace = 19011.1 GFlop/s

//[8] for: Feature = (10, 10), [N, IC, OC] = [64, 512, 512]
//LB = 4: Size = 76.5625, Time = 6.866 msec, Performace = 23946.5 GFlop/s

#endif

//(1) FW = 7
//(2) sh = sw = 1
//(3) GN: (IC    ) % 64
//(4) GM: (N * IH) % 32
//(5) OC % 8 == 0
//(6) IW >= 10
//(7) (pw >= 3) && (OW - IW - pw + 3 >= 0)
#ifndef COMPILE
#define DECONV_3D_WINOGRAD_S16_W7_32X32_P3_TEX
#endif
#ifndef DECONV_3D_WINOGRAD_S16_W7_32X32_P3_TEX
#define DECONV_3D_WINOGRAD_S16_W7_32X32_P3_TEX

template<int FH>
inline void deconv3D_dX_Winograd_s16_W7_32x32_p3_tex(jlong* streams, int &index, int length,
	const float*  deltaY, cudaTextureObject_t texDy, int OH, int OW,
	const float*       W,
	      float*  deltaX, int IH, int IW,
	int N, int IC, int OC,
	int ph, int pw,
	int GN, int GM, int &GMr, int &GNr)
{
	//======[Stage1 (IW % 10): Winograd_F(10, 7)]=========================================================================
	next_cudaStream(stream, streams, index, length);
#ifdef ENABLE_DECONV3D_WINOGRAD_FAX7_CHANNEL_TEMPLATE
	//IC = 2^x
	if      (IC ==  32 && OC ==  32) winograd_f10x7_k32x320_p3_CT_tex(stream, texDy, OH, OW, W, FH, deltaX, IH, IW, N,  32,  32, ph, pw);
	else if (IC ==  64 && OC ==  64) winograd_f10x7_k32x320_p3_CT_tex(stream, texDy, OH, OW, W, FH, deltaX, IH, IW, N,  64,  64, ph, pw);
	else if (IC == 128 && OC == 128) winograd_f10x7_k32x320_p3_CT_tex(stream, texDy, OH, OW, W, FH, deltaX, IH, IW, N, 128, 128, ph, pw);
	else if (IC == 256 && OC == 256) winograd_f10x7_k32x320_p3_CT_tex(stream, texDy, OH, OW, W, FH, deltaX, IH, IW, N, 256, 256, ph, pw);
	else if (IC == 512 && OC == 512) winograd_f10x7_k32x320_p3_CT_tex(stream, texDy, OH, OW, W, FH, deltaX, IH, IW, N, 512, 512, ph, pw);
	//IC = 64x
	else if (IC == 192 && OC == 192) winograd_f10x7_k32x320_p3_CT_tex(stream, texDy, OH, OW, W, FH, deltaX, IH, IW, N, 192, 192, ph, pw);
	else if (IC == 320 && OC == 320) winograd_f10x7_k32x320_p3_CT_tex(stream, texDy, OH, OW, W, FH, deltaX, IH, IW, N, 320, 320, ph, pw);
	else if (IC == 384 && OC == 384) winograd_f10x7_k32x320_p3_CT_tex(stream, texDy, OH, OW, W, FH, deltaX, IH, IW, N, 384, 384, ph, pw);
	else if (IC == 448 && OC == 448) winograd_f10x7_k32x320_p3_CT_tex(stream, texDy, OH, OW, W, FH, deltaX, IH, IW, N, 448, 448, ph, pw);
	else
#endif
	winograd_f10x7_k32x320_p3_tex(stream, texDy, OH, OW, W, FH, deltaX, IH, IW, N, IC, OC, ph, pw);//common
	int IWr = IW % 10;

	//======[Stage3 (IW % 2): Winograd_F(2 + 2, 7) ]======================================================================
	if ((IWr >= 4) && !(IC & 63)) {//Remainder: 4, 5, 6, 7, 8, 9
		const int iw_index = IW - IWr;
		next_cudaStream(stream1, streams, index, length);
#ifdef ENABLE_DECONV3D_WINOGRAD_FAX7_CHANNEL_TEMPLATE
		//IC = 2^x
		if      (IC ==  64 && OC ==  64) winograd_f2x7_ruse_k64x64C_p3_CT_tex(stream1, iw_index, texDy, OH, OW, W, FH, deltaX, IH, IW, N,  64,  64, ph, pw, IWr);
		else if (IC == 128 && OC == 128) winograd_f2x7_ruse_k64x64C_p3_CT_tex(stream1, iw_index, texDy, OH, OW, W, FH, deltaX, IH, IW, N, 128, 128, ph, pw, IWr);
		else if (IC == 256 && OC == 256) winograd_f2x7_ruse_k64x64C_p3_CT_tex(stream1, iw_index, texDy, OH, OW, W, FH, deltaX, IH, IW, N, 256, 256, ph, pw, IWr);
		else if (IC == 512 && OC == 512) winograd_f2x7_ruse_k64x64C_p3_CT_tex(stream1, iw_index, texDy, OH, OW, W, FH, deltaX, IH, IW, N, 512, 512, ph, pw, IWr);
		//IC = 64x
		else if (IC == 192 && OC == 192) winograd_f2x7_ruse_k64x64C_p3_CT_tex(stream1, iw_index, texDy, OH, OW, W, FH, deltaX, IH, IW, N, 192, 192, ph, pw, IWr);
		else if (IC == 320 && OC == 320) winograd_f2x7_ruse_k64x64C_p3_CT_tex(stream1, iw_index, texDy, OH, OW, W, FH, deltaX, IH, IW, N, 320, 320, ph, pw, IWr);
		else if (IC == 384 && OC == 384) winograd_f2x7_ruse_k64x64C_p3_CT_tex(stream1, iw_index, texDy, OH, OW, W, FH, deltaX, IH, IW, N, 384, 384, ph, pw, IWr);
		else if (IC == 448 && OC == 448) winograd_f2x7_ruse_k64x64C_p3_CT_tex(stream1, iw_index, texDy, OH, OW, W, FH, deltaX, IH, IW, N, 448, 448, ph, pw, IWr);
		else 
#endif
		winograd_f2x7_ruse_k64x64C_p3_tex(stream1, iw_index, texDy, OH, OW, W, FH, deltaX, IH, IW, N, IC, OC, ph, pw, IWr);//common
		IWr = IWr & 3;//IWr % 4;
	}	

	//======[Stage3 (IW % 2): Winograd_F(2, 7) ]==========================================================================
	if ((IWr >= 2) && !(IC & 63)) {//Remainder: 1, 2, 3
		const int iw_index = IW - IWr;
		next_cudaStream(stream2, streams, index, length);
#ifdef ENABLE_DECONV3D_WINOGRAD_FAX7_CHANNEL_TEMPLATE
		//IC = 2^x
		if      (IC ==  64 && OC ==  64) winograd_f2x7_k64x64C_CT_tex(stream2, iw_index, texDy, OH, OW, W, FH, deltaX, IH, IW, N,  64,  64, ph, pw, IWr);
		else if (IC == 128 && OC == 128) winograd_f2x7_k64x64C_CT_tex(stream2, iw_index, texDy, OH, OW, W, FH, deltaX, IH, IW, N, 128, 128, ph, pw, IWr);
		else if (IC == 256 && OC == 256) winograd_f2x7_k64x64C_CT_tex(stream2, iw_index, texDy, OH, OW, W, FH, deltaX, IH, IW, N, 256, 256, ph, pw, IWr);
		else if (IC == 512 && OC == 512) winograd_f2x7_k64x64C_CT_tex(stream2, iw_index, texDy, OH, OW, W, FH, deltaX, IH, IW, N, 512, 512, ph, pw, IWr);
		//IC = 64x
		else if (IC == 192 && OC == 192) winograd_f2x7_k64x64C_CT_tex(stream2, iw_index, texDy, OH, OW, W, FH, deltaX, IH, IW, N, 192, 192, ph, pw, IWr);
		else if (IC == 320 && OC == 320) winograd_f2x7_k64x64C_CT_tex(stream2, iw_index, texDy, OH, OW, W, FH, deltaX, IH, IW, N, 320, 320, ph, pw, IWr);
		else if (IC == 384 && OC == 384) winograd_f2x7_k64x64C_CT_tex(stream2, iw_index, texDy, OH, OW, W, FH, deltaX, IH, IW, N, 384, 384, ph, pw, IWr);
		else if (IC == 448 && OC == 448) winograd_f2x7_k64x64C_CT_tex(stream2, iw_index, texDy, OH, OW, W, FH, deltaX, IH, IW, N, 448, 448, ph, pw, IWr);
		else 
#endif
		winograd_f2x7_k64x64C_tex(stream2, iw_index, texDy, OH, OW, W, FH, deltaX, IH, IW, N, IC, OC, ph, pw, IWr);//common
		IWr = IWr & 1;//IWr % 2;
	}

	//======[Stage4: GEMM]================================================================================================
	if (IWr > 0) {//Remainder: 1
		const int GN32 = ((IC  ) >> 5 << 5);      //GN   = 32x
		const int GM32 = ((N*IH) >> 5 << 5) * IWr;//N*IH = 32x
		index = 0;//save L2 cache: texDy != deltaY
		deconv3D_dX_ZeroPadding_s1_32x32C(streams, index, length,
			deltaY, OH, OW, W, FH, 7, deltaX, IH, IW, IWr, N, IC, OC, ph, pw,
			GN32, GM32, 0, 0, IW - IWr);
	}

	//=====[Process GNr and GMr]=============================================================================
	GNr = GN & 31;         //32 channels
	GMr = ((N*IH) & 31)*OW;//32 groups
}

#endif

#endif


//FW == 7
#ifndef DECONV_3D_WINOGRAD_S16_W7_64X32_P3_TEXTURE
#define DECONV_3D_WINOGRAD_S16_W7_64X32_P3_TEXTURE

#ifndef DECONV_3D_WINOGRAD_S16_W7_64X32_P3_TEXTURE_BENCH_MARK
#define DECONV_3D_WINOGRAD_S16_W7_64X32_P3_TEXTURE_BENCH_MARK

//[0] for: Feature = (128, 128), [N, IC, OC] = [32, 64, 64]
//LB = 4: Size = 98, Time = 7.401 msec, Performace = 28435.8 GFlop/s

//[1] for: Feature = (120, 120), [N, IC, OC] = [32, 64, 64]
//LB = 4: Size = 86.1328, Time = 6.021 msec, Performace = 30720.6 GFlop/s

//[2] for: Feature = (80, 80), [N, IC, OC] = [64, 64, 64]
//LB = 4: Size = 76.5625, Time = 5.287 msec, Performace = 31098.3 GFlop/s

//[3] for: Feature = (64, 64), [N, IC, OC] = [64, 64, 64]
//LB = 4: Size = 49, Time = 3.665 msec, Performace = 28711.2 GFlop/s

//[4] for: Feature = (40, 40), [N, IC, OC] = [64, 128, 128]
//LB = 4: Size = 76.5625, Time = 5.857 msec, Performace = 28071.8 GFlop/s

//[5] for: Feature = (32, 32), [N, IC, OC] = [64, 128, 128]
//LB = 4: Size = 98, Time = 7.716 msec, Performace = 27274.9 GFlop/s

//[6] for: Feature = (20, 20), [N, IC, OC] = [64, 256, 256]
//LB = 4:  Size = 76.5625, Time = 5.659 msec, Performace = 29054   GFlop/s

//[7] for: Feature = (16, 16), [N, IC, OC] = [64, 256, 256]
//LB = 4: Size = 98, Time = 10.127 msec, Performace = 20781.4 GFlop/s

//[8] for: Feature = (10, 10), [N, IC, OC] = [64, 512, 512]
//LB = 4: Size = 76.5625, Time = 5.863 msec, Performace = 28043.1 GFlop/s

#endif

//(1) FW = 7
//(2) sh = sw = 1
//(3) GN: (IC    ) % 64
//(4) GM: (N * IH) % 32
//(5) OC % 8 == 0
//(6) IW >= 10
//(7) (pw >= 3) && (OW - IW - pw + 3 >= 0)
#ifndef COMPILE
#define DECONV_3D_WINOGRAD_S16_W7_64X32_P3_TEX
#endif
#ifndef DECONV_3D_WINOGRAD_S16_W7_64X32_P3_TEX
#define DECONV_3D_WINOGRAD_S16_W7_64X32_P3_TEX

template<int FH>
inline void deconv3D_dX_Winograd_s16_W7_64x32_p3_tex(jlong* streams, int &index, int length,
	const float*  deltaY, cudaTextureObject_t texDy, int OH, int OW,
	const float*       W,
	      float*  deltaX, int IH, int IW,
	int N, int IC, int OC,
	int ph, int pw,
	int GN, int GM, int &GMr, int &GNr)
{
	//======[Stage1 (IW % 10): Winograd_F(10, 7)]=========================================================================
	next_cudaStream(stream, streams, index, length);
#ifdef ENABLE_DECONV3D_WINOGRAD_FAX7_CHANNEL_TEMPLATE
	//IC = 2^x
	if      (OC ==  32) winograd_f10x7_k64x320_p3_OCT_tex(stream, texDy, OH, OW, W, FH, deltaX, IH, IW, N, IC,  32, ph, pw);
	else if (OC ==  64) winograd_f10x7_k64x320_p3_OCT_tex(stream, texDy, OH, OW, W, FH, deltaX, IH, IW, N, IC,  64, ph, pw);
	else if (OC == 128) winograd_f10x7_k64x320_p3_OCT_tex(stream, texDy, OH, OW, W, FH, deltaX, IH, IW, N, IC, 128, ph, pw);
	else if (OC == 256) winograd_f10x7_k64x320_p3_OCT_tex(stream, texDy, OH, OW, W, FH, deltaX, IH, IW, N, IC, 256, ph, pw);
	else if (OC == 512) winograd_f10x7_k64x320_p3_OCT_tex(stream, texDy, OH, OW, W, FH, deltaX, IH, IW, N, IC, 512, ph, pw);
	//IC = 64x
	else if (OC == 192) winograd_f10x7_k64x320_p3_OCT_tex(stream, texDy, OH, OW, W, FH, deltaX, IH, IW, N, IC, 192, ph, pw);
	else if (OC == 320) winograd_f10x7_k64x320_p3_OCT_tex(stream, texDy, OH, OW, W, FH, deltaX, IH, IW, N, IC, 320, ph, pw);
	else if (OC == 384) winograd_f10x7_k64x320_p3_OCT_tex(stream, texDy, OH, OW, W, FH, deltaX, IH, IW, N, IC, 384, ph, pw);
	else if (OC == 448) winograd_f10x7_k64x320_p3_OCT_tex(stream, texDy, OH, OW, W, FH, deltaX, IH, IW, N, IC, 448, ph, pw);
	else
#endif
	winograd_f10x7_k64x320_p3_tex(stream, texDy, OH, OW, W, FH, deltaX, IH, IW, N, IC, OC, ph, pw);//common
	int IWr = IW % 10;

	//======[Stage3 (IW % 2): Winograd_F(2 + 2, 7) ]======================================================================
	if (IWr >= 4) {//Remainder: 4, 5, 6, 7, 8, 9
		const int iw_index = IW - IWr;
		next_cudaStream(stream1, streams, index, length);
#ifdef ENABLE_DECONV3D_WINOGRAD_FAX7_CHANNEL_TEMPLATE
		//IC = 2^x
		if      (IC ==  64 && OC ==  64) winograd_f2x7_ruse_k64x64C_p3_CT_tex(stream1, iw_index, texDy, OH, OW, W, FH, deltaX, IH, IW, N,  64,  64, ph, pw, IWr);
		else if (IC == 128 && OC == 128) winograd_f2x7_ruse_k64x64C_p3_CT_tex(stream1, iw_index, texDy, OH, OW, W, FH, deltaX, IH, IW, N, 128, 128, ph, pw, IWr);
		else if (IC == 256 && OC == 256) winograd_f2x7_ruse_k64x64C_p3_CT_tex(stream1, iw_index, texDy, OH, OW, W, FH, deltaX, IH, IW, N, 256, 256, ph, pw, IWr);
		else if (IC == 512 && OC == 512) winograd_f2x7_ruse_k64x64C_p3_CT_tex(stream1, iw_index, texDy, OH, OW, W, FH, deltaX, IH, IW, N, 512, 512, ph, pw, IWr);
		//IC = 64x
		else if (IC == 192 && OC == 192) winograd_f2x7_ruse_k64x64C_p3_CT_tex(stream1, iw_index, texDy, OH, OW, W, FH, deltaX, IH, IW, N, 192, 192, ph, pw, IWr);
		else if (IC == 320 && OC == 320) winograd_f2x7_ruse_k64x64C_p3_CT_tex(stream1, iw_index, texDy, OH, OW, W, FH, deltaX, IH, IW, N, 320, 320, ph, pw, IWr);
		else if (IC == 384 && OC == 384) winograd_f2x7_ruse_k64x64C_p3_CT_tex(stream1, iw_index, texDy, OH, OW, W, FH, deltaX, IH, IW, N, 384, 384, ph, pw, IWr);
		else if (IC == 448 && OC == 448) winograd_f2x7_ruse_k64x64C_p3_CT_tex(stream1, iw_index, texDy, OH, OW, W, FH, deltaX, IH, IW, N, 448, 448, ph, pw, IWr);
		else 
#endif
		winograd_f2x7_ruse_k64x64C_p3_tex(stream1, iw_index, texDy, OH, OW, W, FH, deltaX, IH, IW, N, IC, OC, ph, pw, IWr);//common
		IWr = IWr & 3;//IWr % 4;
	}	

	//======[Stage3 (IW % 2): Winograd_F(2, 7) ]==========================================================================
	if (IWr >= 2) {//Remainder: 2, 3
		const int iw_index = IW - IWr;
		next_cudaStream(stream2, streams, index, length);
#ifdef ENABLE_DECONV3D_WINOGRAD_FAX7_CHANNEL_TEMPLATE
		//IC = 2^x
		if      (IC ==  64 && OC ==  64) winograd_f2x7_k64x64C_CT_tex(stream2, iw_index, texDy, OH, OW, W, FH, deltaX, IH, IW, N,  64,  64, ph, pw, IWr);
		else if (IC == 128 && OC == 128) winograd_f2x7_k64x64C_CT_tex(stream2, iw_index, texDy, OH, OW, W, FH, deltaX, IH, IW, N, 128, 128, ph, pw, IWr);
		else if (IC == 256 && OC == 256) winograd_f2x7_k64x64C_CT_tex(stream2, iw_index, texDy, OH, OW, W, FH, deltaX, IH, IW, N, 256, 256, ph, pw, IWr);
		else if (IC == 512 && OC == 512) winograd_f2x7_k64x64C_CT_tex(stream2, iw_index, texDy, OH, OW, W, FH, deltaX, IH, IW, N, 512, 512, ph, pw, IWr);
		//IC = 64x
		else if (IC == 192 && OC == 192) winograd_f2x7_k64x64C_CT_tex(stream2, iw_index, texDy, OH, OW, W, FH, deltaX, IH, IW, N, 192, 192, ph, pw, IWr);
		else if (IC == 320 && OC == 320) winograd_f2x7_k64x64C_CT_tex(stream2, iw_index, texDy, OH, OW, W, FH, deltaX, IH, IW, N, 320, 320, ph, pw, IWr);
		else if (IC == 384 && OC == 384) winograd_f2x7_k64x64C_CT_tex(stream2, iw_index, texDy, OH, OW, W, FH, deltaX, IH, IW, N, 384, 384, ph, pw, IWr);
		else if (IC == 448 && OC == 448) winograd_f2x7_k64x64C_CT_tex(stream2, iw_index, texDy, OH, OW, W, FH, deltaX, IH, IW, N, 448, 448, ph, pw, IWr);
		else 
#endif
		winograd_f2x7_k64x64C_tex(stream2, iw_index, texDy, OH, OW, W, FH, deltaX, IH, IW, N, IC, OC, ph, pw, IWr);//common
		IWr = IWr & 1;//IWr % 2;
	}

	//======[Stage4: GEMM]================================================================================================
	if (IWr > 0) {//Remainder: 1
		const int GN64 = ((IC  ) >> 6 << 6);      //GN   = 64x
		const int GM32 = ((N*IH) >> 5 << 5) * IWr;//N*IH = 32x
		index = 0;//save L2 cache: texDy != deltaY
		deconv3D_dX_ZeroPadding_s1_32x32C(streams, index, length,
			deltaY, OH, OW, W, FH, 7, deltaX, IH, IW, IWr, N, IC, OC, ph, pw,
			GN64, GM32, 0, 0, IW - IWr);
	}

	//=====[Process GNr and GMr]=============================================================================
	GNr = GN & 63;         //64 channels
	GMr = ((N*IH) & 31)*OW;//32 groups
}

#endif

#endif


//FW == 8
#ifndef DECONV_3D_WINOGRAD_S16_W8_32X32_P4_TEXTURE
#define DECONV_3D_WINOGRAD_S16_W8_32X32_P4_TEXTURE

#ifndef DECONV_3D_WINOGRAD_S16_W8_32X32_P4_TEXTURE_BENCH_MARK
#define DECONV_3D_WINOGRAD_S16_W8_32X32_P4_TEXTURE_BENCH_MARK

//[0] for: Feature = (128, 128), [N, IC, OC] = [32, 64, 64]
//LB   = 4: Size = 128, Time = 10.082 msec, Performace = 27264.2 GFlop/s
//ruse = 4: Size = 128, Time = 10.082 msec, Performace = 27264.2 GFlop/s

//[1] for: Feature = (72, 72), [N, IC, OC] = [64, 64, 64]
//LB =   4: Size = 81, Time = 6.076 msec, Performace = 28628.4 GFlop/s
//ruse = 4: Size = 81, Time = 5.97 msec, Performace = 29136.7 GFlop/s

//[2] for: Feature = (64, 64), [N, IC, OC] = [128, 64, 64]
//LB = 4: Size = 128, Time = 9.946 msec, Performace = 27637 GFlop/s

//[3] for: Feature = (56, 56), [N, IC, OC] = [128, 64, 64]
//LB   = 4: Size = 98, Time = 7.942 msec, Performace = 26498.8 GFlop/s
//ruse = 4: Size = 98, Time = 7.82  msec, Performace = 26912.2 GFlop/s

//[4] for: Feature = (36, 36), [N, IC, OC] = [64, 128, 128]
//LB   = 4: Size = 81, Time = 7.268 msec, Performace = 23933.2 GFlop/s
//ruse = 4: Size = 81, Time = 6.912 msec, Performace = 25165.8 GFlop/s

//[5] for: Feature = (32, 32), [N, IC, OC] = [128, 128, 128]
//LB   = 4: Size = 128, Time = 12.842 msec, Performace = 21404.6 GFlop/s

//[6] for: Feature = (28, 28), [N, IC, OC] = [128, 128, 128]
//LB = 4: Size = 98, Time = 9.938 msec, Performace = 21176.6 GFlop/s

//[7] for: Feature = (18, 18), [N, IC, OC] = [64, 256, 256]
//LB   = 4: Size = 81, Time = 6.664 msec, Performace = 26102.4 GFlop/s
//ruse = 4: Size = 81, Time = 6.226 msec, Performace = 27938.7 GFlop/s

#endif

//(1) FW = 8
//(2) sh = sw = 1
//(3) GN: (IC    ) % 64
//(4) GM: (N * IH) % 32
//(5) OC % 8 == 0
//(6) IW >= 9
//(7) (pw >= 3) && (OW - IW - pw + 5 >= 0)
#ifndef COMPILE
#define DECONV_3D_WINOGRAD_S16_W8_32X32_P4_TEX
#endif
#ifndef DECONV_3D_WINOGRAD_S16_W8_32X32_P4_TEX
#define DECONV_3D_WINOGRAD_S16_W8_32X32_P4_TEX

template<int FH>
inline void deconv3D_dX_Winograd_s16_W8_32x32_p3_tex(jlong* streams, int &index, int length,
	const float*  deltaY, cudaTextureObject_t texDy, int OH, int OW,
	const float*       W,
	      float*  deltaX, int IH, int IW,
	int N, int IC, int OC,
	int ph, int pw,
	int GN, int GM, int &GMr, int &GNr)
{
	//======[Stage1 (IW % 9): Winograd_F(9, 8)]=============================================================
	next_cudaStream(stream, streams, index, length);
	if ((IW >= 18) && ((IW % 18) == (IW % 9))) {//IW % 18 == IW % 9 -> IWr <= 8
		winograd_f9x8_ruse_k32x288_p3_tex(stream, texDy, OH, OW, W, FH, deltaX, IH, IW, N, IC, OC, ph, pw);//common
	}
#ifdef ENABLE_DECONV3D_WINOGRAD_F9X8_CHANNEL_TEMPLATE
	//IC = 2^x
	else if (IC ==  32) { if (OC ==  32) winograd_f9x8_k32x288_p3_CT_tex(stream, texDy, OH, OW, W, FH, deltaX, IH, IW, N,  32,  32, ph, pw); else winograd_f9x8_k32x288_p3_ICT_tex(stream, texDy, OH, OW, W, FH, deltaX, IH, IW, N,  32, OC, ph, pw); }
	else if (IC ==  64) { if (OC ==  64) winograd_f9x8_k32x288_p3_CT_tex(stream, texDy, OH, OW, W, FH, deltaX, IH, IW, N,  64,  64, ph, pw); else winograd_f9x8_k32x288_p3_ICT_tex(stream, texDy, OH, OW, W, FH, deltaX, IH, IW, N,  64, OC, ph, pw); }
	else if (IC == 128) { if (OC == 128) winograd_f9x8_k32x288_p3_CT_tex(stream, texDy, OH, OW, W, FH, deltaX, IH, IW, N, 128, 128, ph, pw); else winograd_f9x8_k32x288_p3_ICT_tex(stream, texDy, OH, OW, W, FH, deltaX, IH, IW, N, 128, OC, ph, pw); }
	else if (IC == 256) { if (OC == 256) winograd_f9x8_k32x288_p3_CT_tex(stream, texDy, OH, OW, W, FH, deltaX, IH, IW, N, 256, 256, ph, pw); else winograd_f9x8_k32x288_p3_ICT_tex(stream, texDy, OH, OW, W, FH, deltaX, IH, IW, N, 256, OC, ph, pw); }
	else if (IC == 512) { if (OC == 512) winograd_f9x8_k32x288_p3_CT_tex(stream, texDy, OH, OW, W, FH, deltaX, IH, IW, N, 512, 512, ph, pw); else winograd_f9x8_k32x288_p3_ICT_tex(stream, texDy, OH, OW, W, FH, deltaX, IH, IW, N, 512, OC, ph, pw); }
	//IC = 32x
	else if (IC == 192) { if (OC == 192) winograd_f9x8_k32x288_p3_CT_tex(stream, texDy, OH, OW, W, FH, deltaX, IH, IW, N, 192, 192, ph, pw); else winograd_f9x8_k32x288_p3_ICT_tex(stream, texDy, OH, OW, W, FH, deltaX, IH, IW, N, 192, OC, ph, pw); }
	else if (IC == 224) { if (OC == 224) winograd_f9x8_k32x288_p3_CT_tex(stream, texDy, OH, OW, W, FH, deltaX, IH, IW, N, 224, 224, ph, pw); else winograd_f9x8_k32x288_p3_ICT_tex(stream, texDy, OH, OW, W, FH, deltaX, IH, IW, N, 224, OC, ph, pw); }
	else if (IC == 320) { if (OC == 320) winograd_f9x8_k32x288_p3_CT_tex(stream, texDy, OH, OW, W, FH, deltaX, IH, IW, N, 320, 320, ph, pw); else winograd_f9x8_k32x288_p3_ICT_tex(stream, texDy, OH, OW, W, FH, deltaX, IH, IW, N, 320, OC, ph, pw); }
	else if (IC == 384) { if (OC == 384) winograd_f9x8_k32x288_p3_CT_tex(stream, texDy, OH, OW, W, FH, deltaX, IH, IW, N, 384, 384, ph, pw); else winograd_f9x8_k32x288_p3_ICT_tex(stream, texDy, OH, OW, W, FH, deltaX, IH, IW, N, 384, OC, ph, pw); }
	else if (IC == 448) { if (OC == 448) winograd_f9x8_k32x288_p3_CT_tex(stream, texDy, OH, OW, W, FH, deltaX, IH, IW, N, 448, 448, ph, pw); else winograd_f9x8_k32x288_p3_ICT_tex(stream, texDy, OH, OW, W, FH, deltaX, IH, IW, N, 448, OC, ph, pw); }
#endif
	else winograd_f9x8_k32x288_p3_tex(stream, texDy, OH, OW, W, FH, deltaX, IH, IW, N, IC, OC, ph, pw);//common

	//======[Stage2 (OW % 5): Winograd_F(5, 4)]=============================================================
	int IWr = IW % 9; 
	if ((IWr >= 5) && !(IC & 63)) {//Remainder: 5, 6, 7, 8
		winograd_SFW_f5x4_k64x160C_tex(stream, (IW - IWr), texDy, OH, OW, W, FH, 8, deltaX, IH, IW, N, IC, OC, ph, pw, IWr);
		IWr = IWr % 5;
	}

	//======[Stage3 (OW % 3): Winograd_F(3, 2)]=============================================================
	if ((IWr >= 3) && !((N*IH) & 63) && !(IC & 63)) {//Remainder: 3, 4
		winograd_SFW_f3x2_k64x192C_tex(stream, (IW - IWr), texDy, OH, OW, W, FH, 8, deltaX, IH, IW, IC, OC, ph, pw, IWr);
		IWr = IWr % 3;
	}

	//======[Stage4: GEMM]==================================================================================
	if (IWr > 0) {//Remainder: 1, 2
		const int GN32 = ((IC  ) >> 5 << 5);      //GN   = 32x
		const int GM32 = ((N*IH) >> 5 << 5) * IWr;//N*IH = 32x
		index = 0;//save L2 cache: texDy != deltaY
		deconv3D_dX_ZeroPadding_s1_32x32C(streams, index, length,
			deltaY, OH, OW, W, FH, 8, deltaX, IH, IW, IWr, N, IC, OC, ph, pw,
			GN32, GM32, 0, 0, (IW - IWr));
	}

	//=====[Process GNr and GMr]============================================================================
	GNr = GN & 31;         //32 channels
	GMr = ((N*IH) & 31)*OW;//32 groups
}

#endif

#endif


//FW == 8
#ifndef DECONV_3D_WINOGRAD_S16_W8_64X32_P4_TEXTURE
#define DECONV_3D_WINOGRAD_S16_W8_64X32_P4_TEXTURE

#ifndef DECONV_3D_WINOGRAD_S16_W8_64X32_P4_TEXTURE_BENCH_MARK
#define DECONV_3D_WINOGRAD_S16_W8_64X32_P4_TEXTURE_BENCH_MARK

//[0] for: Feature = (128, 128), [N, IC, OC] = [32, 64, 64]
//LB = 4: Size = 128, Time = 8.612 msec, Performace = 31918 GFlop/s

//[1] for: Feature = (72, 72), [N, IC, OC] = [64, 64, 64]
//LB = 4: Size = 81, Time = 5.108 msec, Performace = 34053.7 GFlop/s

//[2] for: Feature = (64, 64), [N, IC, OC] = [128, 64, 64]
//LB = 4: Size = 128, Time = 8.412 msec, Performace = 32676.9 GFlop/s

//[3] for: Feature = (56, 56), [N, IC, OC] = [128, 64, 64]
//LB = 4: Size = 98, Time = 6.916 msec, Performace = 30429.9 GFlop/s

//[4] for: Feature = (36, 36), [N, IC, OC] = [64, 128, 128]
//LB = 4: Size = 81, Time = 6.074 msec, Performace = 28637.8 GFlop/s

//[5] for: Feature = (32, 32), [N, IC, OC] = [128, 128, 128]
//LB = 4: Size = 128, Time = 10.518 msec, Performace = 26134 GFlop/s

//[6] for: Feature = (28, 28), [N, IC, OC] = [128, 128, 128]
//LB = 4: Size = 98, Time = 7.922 msec, Performace = 26565.7 GFlop/s

//[7] for: Feature = (18, 18), [N, IC, OC] = [64, 256, 256]
//LB = 4: Size = 81, Time = 5.652 msec, Performace = 30776 GFlop/s

//[8] for: Feature = (9, 9), [N, IC, OC] = [64, 512, 512]
//LB = 4: Size = 81, Time = 5.352 msec, Performace = 32501.1 GFlop/s

#endif

//(1) FW = 8
//(2) sh = sw = 1
//(3) GN: (IC    ) % 64
//(4) GM: (N * IH) % 32
//(5) OC % 8 == 0
//(6) IW >= 9
//(7) (pw >= 3) && (OW - IW - pw + 5 >= 0)
#ifndef COMPILE
#define DECONV_3D_WINOGRAD_S16_W8_64X32_P4_TEX
#endif
#ifndef DECONV_3D_WINOGRAD_S16_W8_64X32_P4_TEX
#define DECONV_3D_WINOGRAD_S16_W8_64X32_P4_TEX

template<int FH>
inline void deconv3D_dX_Winograd_s16_W8_64x32_p3_tex(jlong* streams, int &index, int length,
	const float*  deltaY, cudaTextureObject_t texDy, int OH, int OW,
	const float*       W,
	      float*  deltaX, int IH, int IW,
	int N, int IC, int OC,
	int ph, int pw,
	int GN, int GM, int &GMr, int &GNr)
{
	//======[Stage1 (IW % 9): Winograd_F(9, 8)]=============================================================
	next_cudaStream(stream, streams, index, length);
#ifdef ENABLE_DECONV3D_WINOGRAD_F9X8_CHANNEL_TEMPLATE
	//OC = 2^x
	if      (OC ==  64) winograd_f9x8_k64x288_p3_OCT_tex(stream, texDy, OH, OW, W, FH, deltaX, IH, IW, N, IC,  64, ph, pw);
	else if (OC == 128) winograd_f9x8_k64x288_p3_OCT_tex(stream, texDy, OH, OW, W, FH, deltaX, IH, IW, N, IC, 128, ph, pw);
	else if (OC == 256) winograd_f9x8_k64x288_p3_OCT_tex(stream, texDy, OH, OW, W, FH, deltaX, IH, IW, N, IC, 256, ph, pw);
	else if (OC == 512) winograd_f9x8_k64x288_p3_OCT_tex(stream, texDy, OH, OW, W, FH, deltaX, IH, IW, N, IC, 512, ph, pw);
	//OC = 32x
	else if (OC == 192) winograd_f9x8_k64x288_p3_OCT_tex(stream, texDy, OH, OW, W, FH, deltaX, IH, IW, N, IC, 192, ph, pw);
	else if (OC == 224) winograd_f9x8_k64x288_p3_OCT_tex(stream, texDy, OH, OW, W, FH, deltaX, IH, IW, N, IC, 224, ph, pw);
	else if (OC == 320) winograd_f9x8_k64x288_p3_OCT_tex(stream, texDy, OH, OW, W, FH, deltaX, IH, IW, N, IC, 320, ph, pw);
	else if (OC == 384) winograd_f9x8_k64x288_p3_OCT_tex(stream, texDy, OH, OW, W, FH, deltaX, IH, IW, N, IC, 384, ph, pw);
	else if (OC == 448) winograd_f9x8_k64x288_p3_OCT_tex(stream, texDy, OH, OW, W, FH, deltaX, IH, IW, N, IC, 448, ph, pw);
	else
#endif
	winograd_f9x8_k64x288_p3_tex(stream, texDy, OH, OW, W, FH, deltaX, IH, IW, N, IC, OC, ph, pw);//common

	//======[Stage2 (OW % 5): Winograd_F(5, 4)]=============================================================
	int IWr = IW % 9;
	if (IWr >= 5) {//Remainder: 5, 6, 7, 8
		winograd_SFW_f5x4_k64x160C_tex(stream, (IW - IWr), texDy, OH, OW, W, FH, 8, deltaX, IH, IW, N, IC, OC, ph, pw, IWr);
		IWr = IWr % 5;
	}

	//======[Stage3 (OW % 3): Winograd_F(3, 2)]=============================================================
	if ((IWr >= 3) && !((N*IH) & 63)) {//Remainder: 3, 4
		winograd_SFW_f3x2_k64x192C_tex(stream, (IW - IWr), texDy, OH, OW, W, FH, 8, deltaX, IH, IW, IC, OC, ph, pw, IWr);
		IWr = IWr % 3;
	}

	//======[Stage4: GEMM]==================================================================================
	if (IWr > 0) {//Remainder: 1, 2
		const int GN64 = ((IC  ) >> 6 << 6);      //GN   = 64x
		const int GM32 = ((N*IH) >> 5 << 5) * IWr;//N*IH = 32x
		index = 0;//save L2 cache: texDy != deltaY
		deconv3D_dX_ZeroPadding_s1_32x32C(streams, index, length,
			deltaY, OH, OW, W, FH, 8, deltaX, IH, IW, IWr, N, IC, OC, ph, pw,
			GN64, GM32, 0, 0, (IW - IWr));
	}

	//=====[Process GNr and GMr]============================================================================
	GNr = GN & 63;         //64 channels
	GMr = ((N*IH) & 31)*OW;//32 groups
}

#endif

#endif


//FW == 9
#ifndef DECONV_3D_WINOGRAD_S16_W9_32X32_P4_TEXTURE
#define DECONV_3D_WINOGRAD_S16_W9_32X32_P4_TEXTURE

#ifndef DECONV_3D_WINOGRAD_S16_W9_32X32_P4_TEXTURE_BENCH_MARK
#define DECONV_3D_WINOGRAD_S16_W9_32X32_P4_TEXTURE_BENCH_MARK

//[0] for: Feature = (128, 128), [N, IC, OC] = [32, 64, 64]
//ruse = 4: Size = 162, Time = 11.858 msec, Performace = 29338.2 GFlop/s

//[1] for: Feature = (124, 124), [N, IC, OC] = [32, 64, 64]
//ruse = 4: Size = 152.033, Time = 12.234 msec, Performace = 26687 GFlop/s

//[2] for: Feature = (64, 64), [N, IC, OC] = [128, 64, 64]
//ruse = 4: Size = 162, Time = 12.14  msec, Performace = 28656.7 GFlop/s

//[3] for: Feature = (60, 60), [N, IC, OC] = [128, 64, 64]
//ruse = 4: Size = 142.383, Time = 11.756 msec, Performace = 26009.2 GFlop/s

//[4] for: Feature = (32, 32), [N, IC, OC] = [128, 128, 128]
//ruse = 4: Size = 162, Time = 14.175 msec, Performace = 24542.7 GFlop/s

//[5] for: Feature = (28, 28), [N, IC, OC] = [128, 128, 128]
//ruse = 4: Size = 124.031, Time = 12.938 msec, Performace = 20587 GFlop/s

//[6] for: Feature = (16, 16), [N, IC, OC] = [128, 256, 256]
//ruse = 4: Size = 162, Time = 12.8   msec, Performace = 27179.1 GFlop/s

//[7] for: Feature = (8, 8), [N, IC, OC] = [128, 512, 512]
//ruse = 4: Size = 162, Time = 13.642 msec, Performace = 25501.6 GFlop/s

#endif

//(1) FW = 9
//(2) sh = sw = 1
//(3) GN: (IC    ) % 64
//(4) GM: (N * IH) % 32
//(5) OC % 8 == 0
//(6) IW >= 8
//(7) (pw >= 4) && (OW - IW - pw + 4 >= 0)
#ifndef COMPILE
#define DECONV_3D_WINOGRAD_S16_W9_32X32_P4_TEX
#endif
#ifndef DECONV_3D_WINOGRAD_S16_W9_32X32_P4_TEX
#define DECONV_3D_WINOGRAD_S16_W9_32X32_P4_TEX

template<int FH>
inline void deconv3D_dX_Winograd_s16_W9_32x32_p4_tex(jlong* streams, int &index, int length,
	const float*  deltaY, cudaTextureObject_t texDy, int OH, int OW,
	const float*       W,
	      float*  deltaX, int IH, int IW,
	int N, int IC, int OC,
	int ph, int pw,
	int GN, int GM, int &GMr, int &GNr)
{
	//======[Stage1 (IW % 8): Winograd_F(8, 9)]=============================================================
	next_cudaStream(stream, streams, index, length);
	if ((IW >= 16) && ((IW & 15) == (IW & 7))) {//IW % 16 == IW % 8 -> IWr <= 7
		winograd_f8x9_ruse_k32x256_p4_tex(stream, texDy, OH, OW, W, FH, deltaX, IH, IW, N, IC, OC, ph, pw);//common
	}
#ifdef ENABLE_DECONV3D_WINOGRAD_F8X9_CHANNEL_TEMPLATE
	//IC = 2^x
	else if (IC ==  32) { if (OC ==  32) winograd_f8x9_k32x256_p4_CT_tex(stream, texDy, OH, OW, W, FH, deltaX, IH, IW, N,  32,  32, ph, pw); else winograd_f8x9_k32x256_p4_ICT_tex(stream, texDy, OH, OW, W, FH, deltaX, IH, IW, N,  32, OC, ph, pw); }
	else if (IC ==  64) { if (OC ==  64) winograd_f8x9_k32x256_p4_CT_tex(stream, texDy, OH, OW, W, FH, deltaX, IH, IW, N,  64,  64, ph, pw); else winograd_f8x9_k32x256_p4_ICT_tex(stream, texDy, OH, OW, W, FH, deltaX, IH, IW, N,  64, OC, ph, pw); }
	else if (IC == 128) { if (OC == 128) winograd_f8x9_k32x256_p4_CT_tex(stream, texDy, OH, OW, W, FH, deltaX, IH, IW, N, 128, 128, ph, pw); else winograd_f8x9_k32x256_p4_ICT_tex(stream, texDy, OH, OW, W, FH, deltaX, IH, IW, N, 128, OC, ph, pw); }
	else if (IC == 256) { if (OC == 256) winograd_f8x9_k32x256_p4_CT_tex(stream, texDy, OH, OW, W, FH, deltaX, IH, IW, N, 256, 256, ph, pw); else winograd_f8x9_k32x256_p4_ICT_tex(stream, texDy, OH, OW, W, FH, deltaX, IH, IW, N, 256, OC, ph, pw); }
	else if (IC == 512) { if (OC == 512) winograd_f8x9_k32x256_p4_CT_tex(stream, texDy, OH, OW, W, FH, deltaX, IH, IW, N, 512, 512, ph, pw);
		else winograd_f8x9_k32x256_p4_ICT_tex(stream, texDy, OH, OW, W, FH, deltaX, IH, IW, N, 512, OC, ph, pw); }
	//IC = 32x
	else if (IC == 192) { if (OC == 192) winograd_f8x9_k32x256_p4_CT_tex(stream, texDy, OH, OW, W, FH, deltaX, IH, IW, N, 192, 192, ph, pw); else winograd_f8x9_k32x256_p4_ICT_tex(stream, texDy, OH, OW, W, FH, deltaX, IH, IW, N, 192, OC, ph, pw); }
	else if (IC == 224) { if (OC == 224) winograd_f8x9_k32x256_p4_CT_tex(stream, texDy, OH, OW, W, FH, deltaX, IH, IW, N, 192, 192, ph, pw); else winograd_f8x9_k32x256_p4_ICT_tex(stream, texDy, OH, OW, W, FH, deltaX, IH, IW, N, 224, OC, ph, pw); }
	else if (IC == 320) { if (OC == 320) winograd_f8x9_k32x256_p4_CT_tex(stream, texDy, OH, OW, W, FH, deltaX, IH, IW, N, 320, 320, ph, pw); else winograd_f8x9_k32x256_p4_ICT_tex(stream, texDy, OH, OW, W, FH, deltaX, IH, IW, N, 320, OC, ph, pw); }
	else if (IC == 384) { if (OC == 384) winograd_f8x9_k32x256_p4_CT_tex(stream, texDy, OH, OW, W, FH, deltaX, IH, IW, N, 384, 384, ph, pw); else winograd_f8x9_k32x256_p4_ICT_tex(stream, texDy, OH, OW, W, FH, deltaX, IH, IW, N, 384, OC, ph, pw); }
	else if (IC == 448) { if (OC == 448) winograd_f8x9_k32x256_p4_CT_tex(stream, texDy, OH, OW, W, FH, deltaX, IH, IW, N, 448, 448, ph, pw); else winograd_f8x9_k32x256_p4_ICT_tex(stream, texDy, OH, OW, W, FH, deltaX, IH, IW, N, 448, OC, ph, pw); }
#endif
	else winograd_f8x9_k32x256_p4_tex(stream, texDy, OH, OW, W, FH, deltaX, IH, IW, N, IC, OC, ph, pw);//common

	//======[Stage2 (IW % 4): Winograd_F(2+2, 3)]===========================================================
	int IWr = IW & 7;
	if ((IWr >= 4) && !((N*IH) & 31) && !(IC & 63)) {//Remainder: 4, 5, 6, 7
		winograd_SFW_f2x3_k64x128x4C_tex(stream, (IW - IWr), texDy, OH, OW, W, FH, 9, deltaX, IH, IW, IC, OC, ph, pw, IWr);
		IWr = IWr & 3;//IWr % 4
	}

	//======[Stage3 (IW % 2): Winograd_F(2, 3)]=============================================================
	if ((IWr >= 2) && !((N*IH) & 63) && !(IC & 63)) {//Remainder: 2, 3
		winograd_SFW_f2x3_k64x128C_tex(stream, (IW - IWr), texDy, OH, OW, W, FH, 9, deltaX, IH, IW, IC, OC, ph, pw, IWr);
		IWr = IWr & 1;//IWr % 2
	}

	//======[Stage2: GEMM]==================================================================================
	if (IWr > 0) {//Remainder: 1
		const int GN32 = ((IC  ) >> 5 << 5);      //GN   = 32x
		const int GM32 = ((N*IH) >> 5 << 5) * IWr;//N*IH = 32x
		index = 0;//save L2 cache: texDy != deltaY
		deconv3D_dX_ZeroPadding_s1_32x32C(streams, index, length,
			deltaY, OH, OW, W, FH, 9, deltaX, IH, IW, IWr, N, IC, OC, ph, pw,
			GN32, GM32, 0, 0, (IW - IWr));
	}

	//=====[Process GNr and GMr]=============================================================================
	GNr = GN & 31;         //32 channels
	GMr = ((N*IH) & 31)*OW;//32 groups
}

#endif

#endif


//FW == 9
#ifndef DECONV_3D_WINOGRAD_S16_W9_64X32_P4_TEXTURE
#define DECONV_3D_WINOGRAD_S16_W9_64X32_P4_TEXTURE

#ifndef DECONV_3D_WINOGRAD_S16_W9_64X32_P4_TEXTURE_BENCH_MARK
#define DECONV_3D_WINOGRAD_S16_W9_64X32_P4_TEXTURE_BENCH_MARK

//[0] for: Feature = (128, 128), [N, IC, OC] = [32, 64, 64]
//ruse = 4: Size = 162, Time = 10.054 msec, Performace = 34602.4 GFlop/s

//[1] for: Feature = (124, 124), [N, IC, OC] = [32, 64, 64]
//ruse = 4: Size = 152.033, Time = 10.226 msec, Performace = 31927.3 GFlop/s

//[2] for: Feature = (64, 64), [N, IC, OC] = [128, 64, 64]
//ruse = 4: Size = 162, Time = 9.95 msec, Performace = 34964.1 GFlop/s

//[3] for: Feature = (60, 60), [N, IC, OC] = [128, 64, 64]
//ruse = 4: Size = 142.383, Time = 10.364 msec, Performace = 29502.6 GFlop/s

//[4] for: Feature = (32, 32), [N, IC, OC] = [128, 128, 128]
//ruse = 4: Size = 162, Time = 11.95 msec, Performace = 29112.3 GFlop/s

//[5] for: Feature = (28, 28), [N, IC, OC] = [128, 128, 128]
//ruse = 4: ize = 124.031, Time = 10.866 msec, Performace = 24512.7 GFlop/s

//[6] for: Feature = (16, 16), [N, IC, OC] = [128, 256, 256]
//ruse = 4: Size = 162, Time = 11.448 msec, Performace = 30388.9 GFlop/s

//[7] for: Feature = (8, 8), [N, IC, OC] = [128, 512, 512]
//ruse = 4: Size = 162, Time = 10.664 msec, Performace = 32623.1 GFlop/s

#endif

//(1) FW = 9
//(2) sh = sw = 1
//(3) GN: (IC    ) % 64
//(4) GM: (N * IH) % 32
//(5) OC % 8 == 0
//(6) IW >= 8
//(7) (pw >= 4) && (OW - IW - pw + 4 >= 0)
#ifndef COMPILE
#define DECONV_3D_WINOGRAD_S16_W9_64X32_P4_TEX
#endif
#ifndef DECONV_3D_WINOGRAD_S16_W9_64X32_P4_TEX
#define DECONV_3D_WINOGRAD_S16_W9_64X32_P4_TEX

template<int FH>
inline void deconv3D_dX_Winograd_s16_W9_64x32_p4_tex(jlong* streams, int &index, int length,
	const float*  deltaY, cudaTextureObject_t texDy, int OH, int OW,
	const float*       W,
	      float*  deltaX, int IH, int IW,
	int N, int IC, int OC,
	int ph, int pw,
	int GN, int GM, int &GMr, int &GNr)
{
	//======[Stage1 (IW % 8): Winograd_F(8, 9)]=============================================================
	next_cudaStream(stream, streams, index, length);
#ifdef ENABLE_DECONV3D_WINOGRAD_F8X9_CHANNEL_TEMPLATE
	//OC = 2^x
	if      (OC ==  64) winograd_f8x9_k64x256_p4_OCT_tex(stream, texDy, OH, OW, W, FH, deltaX, IH, IW, N, IC,  64, ph, pw);
	else if (OC == 128) winograd_f8x9_k64x256_p4_OCT_tex(stream, texDy, OH, OW, W, FH, deltaX, IH, IW, N, IC, 128, ph, pw);
	else if (OC == 256) winograd_f8x9_k64x256_p4_OCT_tex(stream, texDy, OH, OW, W, FH, deltaX, IH, IW, N, IC, 256, ph, pw);
	else if (OC == 512) winograd_f8x9_k64x256_p4_OCT_tex(stream, texDy, OH, OW, W, FH, deltaX, IH, IW, N, IC, 512, ph, pw);
	//OC = 32x
	else if (OC == 192) winograd_f8x9_k64x256_p4_OCT_tex(stream, texDy, OH, OW, W, FH, deltaX, IH, IW, N, IC, 192, ph, pw);
	else if (OC == 224) winograd_f8x9_k64x256_p4_OCT_tex(stream, texDy, OH, OW, W, FH, deltaX, IH, IW, N, IC, 224, ph, pw);
	else if (OC == 320) winograd_f8x9_k64x256_p4_OCT_tex(stream, texDy, OH, OW, W, FH, deltaX, IH, IW, N, IC, 320, ph, pw);
	else if (OC == 384) winograd_f8x9_k64x256_p4_OCT_tex(stream, texDy, OH, OW, W, FH, deltaX, IH, IW, N, IC, 384, ph, pw);
	else if (OC == 448) winograd_f8x9_k64x256_p4_OCT_tex(stream, texDy, OH, OW, W, FH, deltaX, IH, IW, N, IC, 448, ph, pw);
	else
#endif
	winograd_f8x9_k64x256_p4_tex(stream, texDy, OH, OW, W, FH, deltaX, IH, IW, N, IC, OC, ph, pw);//common

	//======[Stage2 (IW % 4): Winograd_F(2+2, 3)]===========================================================
	int IWr = IW & 7;
	if ((IWr >= 4) && !((N*IH) & 31)) {//Remainder: 4, 5, 6, 7
		winograd_SFW_f2x3_k64x128x4C_tex(stream, (IW - IWr), texDy, OH, OW, W, FH, 9, deltaX, IH, IW, IC, OC, ph, pw, IWr);
		IWr = IWr & 3;//IWr % 4
	}

	//======[Stage3 (IW % 2): Winograd_F(2, 3)]=============================================================
	if ((IWr >= 2) && !((N*IH) & 63)) {//Remainder: 2, 3
		winograd_SFW_f2x3_k64x128C_tex(stream, (IW - IWr), texDy, OH, OW, W, FH, 9, deltaX, IH, IW, IC, OC, ph, pw, IWr);
		IWr = IWr & 1;//IWr % 2
	}

	//======[Stage2: GEMM]==================================================================================
	if (IWr > 0) {//Remainder: 1
		const int GN64 = ((IC  ) >> 6 << 6);      //GN   = 64x
		const int GM32 = ((N*IH) >> 5 << 5) * IWr;//N*IH = 32x
		index = 0;//save L2 cache: texDy != deltaY
		deconv3D_dX_ZeroPadding_s1_32x32C(streams, index, length,
			deltaY, OH, OW, W, FH, 9, deltaX, IH, IW, IWr, N, IC, OC, ph, pw,
			GN64, GM32, 0, 0, (IW - IWr));
	}

	//=====[Process GNr and GMr]=============================================================================
	GNr = GN & 63;         //64 channels
	GMr = ((N*IH) & 31)*OW;//32 groups
}

#endif

#endif

#endif


#endif