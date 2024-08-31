#pragma once

#ifndef TEST_H
#define TEST_H

#ifndef COMPILE//<<<<complie-area--------------------------------------------------

#ifndef UTIL
#define UTIL

float* newRandomFloatVec(int length) {
	float *p = new float[length];
	for (int i = 0; i < length; i++) p[i] = (float)(rand() % 1000) / 1000 + 1;
	return p;
}

float *newDevFloatVec(int length) {
	float *dp = NULL;
	size_t size = sizeof(float)*length;
	cudaError_t error = cudaMalloc((void**)&dp, size); printError(error);
	error = cudaMemset(dp, 0, size); printError(error);
	return dp;
}

float* newDevFloatVec(float *p, int length) {
	float *dp = NULL;
	size_t size = sizeof(float)*length;
	cudaError_t error = cudaMalloc((void**)&dp, size); printError(error);
	error = cudaMemcpy(dp, p, size, cudaMemcpyHostToDevice); printError(error);
	return dp;
}

void println(float *p, int length) {
	for (int i = 0; i < length; i++) printf("%f, ", p[i]);
	printf("\n");
}

float samePercent(float *a, float *b, int length) {
	int sum = 0;
	for (int i = 0; i < length; i++)
	{
		//if (b[i] == 0) cout << "zero: " << i << endl;
		if (a[i] == b[i])
		{
			sum++; continue;
		}
		//float dif = fabs(a[i] - b[i]);
		float dif = fabs((a[i] - b[i]) / (a[i] + b[i]));
		if (dif < 1e-3) sum++;
		else {
			//if (a[i] < b[i]) cout << i << ":" << dif << " " << a[i] << ", " << b[i] << endl;
			//cout << i << ":" << dif << " " << a[i] << ", " << b[i] << endl;
		}
		
	}
	return 1.0f*sum / length;
}

float relative_difference(float *a, float *b, int length) {
	double sum = 0;
	int count = 0;
	for (int i = 0; i < length; i++) {
		float abs1 = fabsf(a[i]);
		float abs2 = fabsf(b[i]);
		float dif = (abs1 - abs2) / abs1;
		dif = abs(dif);
		if (dif < 1e-6f) {
			count++;
			//printf("%f, %f, %e\n", abs1, abs2, dif);
		}
		sum += dif;
	}
	printf("count = %d\n", count);
	printf("length = %d\n", length);
	printf("sum = %f\n", sum);
	return sum / length;
}

#include <map>
void relative_dif_dis(float *a, float *b, int length) {
	map<int, int> mp;
	map<int, int>::iterator iter;

	for (int i = 0; i < length; i++) {
		float abs1 = fabsf(a[i]);
		float abs2 = fabsf(b[i]);
		float dif = (abs1 - abs2) / abs1;
		dif = abs(dif);
		int index = dif / (1e-7f);

		iter = mp.find(index);
		if (iter == mp.end()) mp[index] = 1;
		else mp[index]++;
	}

	for (iter = mp.begin(); iter != mp.end(); iter++) {
		cout << iter->first << ", " << iter->second << endl;
	}
}

float samePercent4D(float *A, float *B, int N, int IH, int IW, int IC) {
	int sum = 0;
	for (int n = 0; n < N; n++)
	for (int ih = 0; ih < IH; ih++)
	for (int iw = 0; iw < IW; iw++)
	for (int ic = 0; ic < IC; ic++)
	{
		const int offset = ((n*IH + ih)*IW + iw)*IC + ic;
		const float a = A[offset];
		const float b = B[offset];
		if (a == b) { 
			sum++;
			//printf("%f, %f, [%d, %d, %d, %d]\n", a, b, n, ih, iw, ic);
			continue; 
		}
		float dif = fabs((a - b) / (a + b));
		if (dif < 1e-4) sum++;
		else {
			//printf("%f, %f, [%d, %d, %d, %d]\n", a, b, n, ih, iw, ic);
			//if (a[i] < b[i]) cout << i << ":" << dif << " " << a[i] << ", " << b[i] << endl;
			//cout << i << ":" << dif << " " << a[i] << ", " << b[i] << endl;
		}

	}

	int length = N * IH*IW*IC;
	return 1.0f*sum / length;
}


float zeroPercent(float *a, int length) {
	int sum = 0;
	for (int i = 0; i < length; i++)
		if (a[i] == 0) sum++;
	return 1.0f*sum / length;
}

float sum(float* a, int length) {
	float r = 0;
	for (int i = 0; i < length; i++) r += a[i];
	return r;
}

void check_zero(float* X, int N, int H, int W, int C) {
	for (int n = 0; n < N; n++)
	for (int h = 0; h < H; h++)
	for (int w = 0; w < W; w++)
	for (int c = 0; c < C; c++) {
		const int xoffset = ((n*H + h)*W + w)*C + c;
		float x = X[xoffset];
		if (x == 0) printf("[%d, %d, %d, %d]\n", n, h, w, c);
	}


	//for (int n = 0; n < N; n++)
	//for (int h = 0; h < H; h++)
	//for (int w = 0; w < W; w++) {
	//	const int xoffset = ((n*H + h)*W + w)*C;
	//	float x = X[xoffset];
	//	if (x == 0) printf("[%d, %d, %d]\n", n, h, w);
	//}
}


#endif


#ifndef CONV_3D_CPU
#define CONV_3D_CPU

void CONV_3D_NAIVE(
	float *X, int IH, int IW,
	float *W, int FH, int FW,
	float *Y, int OH, int OW,
	int N, int IC, int OC,
	int sh, int sw,
	int ph, int pw)
{
	//Y[N, OH, OW, OC] = > C[GN, GM]
	for (int oc = 0; oc < OC; oc++)//OUT channel: use kernel[oc]
	for (int n = 0; n < N; n++)//for each sample   
	{
		int ic_s, ih_s, iw_s, oh, ow;
		for (ih_s = -ph, oh = 0; ih_s <= (IH + ph - FH); ih_s += sh, oh++)//oh < OH
		for (iw_s = -pw, ow = 0; iw_s <= (IW + pw - FW); iw_s += sw, ow++)//ow < OW
		{
			float v = 0;
			for (int fh = 0; fh < FH; fh++)
			for (int fw = 0; fw < FW; fw++)
			for (int ic = 0; ic < IC; ic++)
			{
				int ih = ih_s + fh, iw = iw_s + fw;
				if (ih < 0 || iw < 0 || ih >= IH || iw >= IW) continue;
						v += get4d(X, n, ih, iw, ic, IH, IW, IC)*
						get4d(W, oc, fh, fw, ic, FH, FW, IC);
			}
			get4d(Y, n, oh, ow, oc, OH, OW, OC) = v;
		}
	}
}


void CONV_3D_img2col(
	float *X, int IH, int IW, //X[N , IH, IW, IC] => A[GN, GK]
	float *W, int FH, int FW, //W[OC, KH, KW, IC] => B[GK, GM]
	float *Y, int OH, int OW, //Y[N , OH, OW, OC] => C[GN, GM]
	int N, int IC, int OC,
	int sh, int sw,
	int ph, int pw)
{
	int GN = OC;
	int GM = N * OH * OW;
	int GK = FH * FW * IC;

	for (int i = 0; i < GN; i++)
	{
		int oc = i;
		for (int j = 0; j < GM; j++)
		{
			int n = j / (OH*OW);
			int j_res = j % (OH*OW);
			int oh = j_res / OW;
			int ow = j_res % OW;

			double v = 0;
			//float v = 0;
			for (int k = 0; k < GK; k++)
			{
				int fh = k / (FW*IC);
				int k_res = k % (FW*IC);
				int fw = k_res / IC;
				int ic = k_res % IC;
				int ih = oh * sh - ph + fh;
				int iw = ow * sw - pw + fw;

				if (ih < 0 || iw < 0 || ih >= IH || iw >= IW) continue;
				v += get4d(X, n, ih, iw, ic, IH, IW, IC)*
					get4d(W, oc, fh, fw, ic, FH, FW, IC);
			}
			get4d(Y, n, oh, ow, oc, OH, OW, OC) = v;
		}
	}
}

#endif


#ifndef PROOF1
#define PROOF1

//int OH = (IH + 2 * ph - FH) / sh + 1;
//int OW = (IW + 2 * pw - FW) / sw + 1;
void proof1(
	int IH, int IW,
	int FH, int FW,
	int N, int IC, int OC,
	int sh, int sw,
	int ph, int pw)
{
	float *X = newRandomFloatVec(N*IH*IW*IC);
	float *W = newRandomFloatVec(OC*FH*FW*IC);

	int OH = (IH + 2 * ph - FH) / sh + 1;
	int OW = (IW + 2 * pw - FW) / sw + 1;
	printf("(OH, OW) = (%d, %d)\n", OH, OW);

	int sizeY = N * OC*OH*OW;
	float *Y1 = new float[sizeY];
	memset(Y1, 0, sizeof(float)*sizeY);
	float *Y2 = new float[sizeY];
	memset(Y2, 0, sizeof(float)*sizeY);

	//use img2col---------------------------
	CONV_3D_img2col(X, IH, IW,
		W, FH, FW,
		Y1, OH, OW,
		N, IC, OC, sh, sw, ph, pw);
	cout << "use img2col method:"; println(Y1, 10);

	//use naive method----------------------
	CONV_3D_NAIVE(X, IH, IW,
		W, FH, FW,
		Y2, OH, OW,
		N, IC, OC, sh, sw, ph, pw);
	cout << "use naive method:  "; println(Y2, 10);

	float sp = samePercent(Y1, Y2, sizeY);
	cout << "SamePercent: " << sp << endl;
}

//(correct)
void proof()
{
	//int OH = (IH + 2 * ph - FH) / sh + 1;
	//int OW = (IW + 2 * pw - FW) / sw + 1;
	int IH = 64, IW = 64;
	int FH = 8, FW = 8;
	int N = 4;
	int IC = 3, OC = 2;
	int sh = 4, sw = 4, ph = 4, pw = 4;
	proof1(IH, IW, FH, FW, N, IC, OC, sh, sw, ph, pw);
}

#endif


void set_L2cache() {
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);

	size_t size = (prop.l2CacheSize * 0.75);
	if (size > prop.persistingL2CacheMaxSize) size = prop.persistingL2CacheMaxSize;
	cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, size);
}

//#define conv3dGemm_k88R4W3_ic2pow_texV(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, LIC, OC, sh, sw, ph, pw, GN, GM) \
//	conv3dGemm_kernel_8_8R4W3_IC2pow_texture<LB, (1<<LB>>1)>\
//		<<< dim3(1, 65535), dim3(8, 8), 0, stream >>>\
//			(X,IH,IW, CW, Y,(OH*OW),OW, LIC,OC, sh,sw,ph,pw,\
//			oc_index, j_index)

//2D dim3: max = 65535

template<int LB>
void testCorrect(
	int IH, int IW,
	int OH, int OW,
	int FH, int FW,
	int N, int IC, int OC,
	int sh, int sw, int ph, int pw)
{
	if(OH == -1) OH = (IH + 2 * ph - FH) / sh + 1;
	if(OW == -1) OW = (IW + 2 * pw - FW) / sw + 1;

	printf("Test Correct:\n");
	printf("\t(IH, IW) = (%d, %d)\n", IH, IW);
	printf("\t(FH, FW) = (%d, %d)\n", FH, FW);
	printf("\t(N, IC, OC) = (%d, %d, %d)\n", N, IC, OC);
	printf("\t(OH, OW) = (%d, %d)\n", OH, OW);
	printf("\t(sh, sw, ph, pw) = (%d, %d, %d, %d)\n", sh, sw, ph, pw);
	const int GN = OC;
	const int GM = N * OH*OW;
	const int GK = FH * FW*IC;
	printf("\t(GN, GM, GK) = (%d, %d, %d)\n", GN, GM, GK);

	int sizeX = N * IH*IW*IC;
	int sizeY = N * OH*OW*OC;
	int sizeW = OC * FH*FW*IC;

	float *X = newRandomFloatVec(sizeX);
	float *W = newRandomFloatVec(sizeW);
	float *Y1 = new float[sizeY]; memset(Y1, 0, sizeof(float)*sizeY);
	float *Y2 = new float[sizeY]; memset(Y2, 0, sizeof(float)*sizeY);
	float *Y3 = new float[sizeY]; memset(Y3, 0, sizeof(float)*sizeY);

	//CPU----------------------------
	//CONV_3D_NAIVE(X, IH, IW, W, FH, FW, Y1, OH, OW, N, IC, OC, sh, sw, ph, pw); printf("CPU1: "); println(Y1, 10);
	CONV_3D_img2col(X, IH, IW, W, FH, FW, Y1, OH, OW, N, IC, OC, sh, sw, ph, pw); printf("CPU2: "); println(Y1, 10);

	//GPU-----------------------------
	float *dX = newDevFloatVec(X, sizeX);
	float *dW = newDevFloatVec(W, sizeW);
	float *dY = newDevFloatVec(sizeY);

	int length = 8;
	jlong *streams = new jlong[8];
	for (int i = 0; i < 8; i++) {
		cudaStream_t stream;
		cudaStreamCreate(&stream);
		streams[i] = (intptr_t)stream;
	}

	cudaError_t error;
	
	float* dCW = newDevFloatVec(sizeW);
	float* dG = newDevFloatVec(OC * IC * 4 * 4);
	cudaTextureObject_t texX = floatTexture(dX, sizeX);

	int DH = 0, DW = 0; float* dD = NULL;
	if (FH == 3 && FW == 3) {
		DH = OH - 1 + FH;
		DW = OW - 1 + FW;
		dD = newDevFloatVec(IC * N * DH * DW);
	}

	if (FH == 2 && FW == 2) {
		DH = (OH + 2) / 3;
		DW = (OW + 2) / 3;
	}

	float Q = PADDING_SCALE_UP(IH, IW, OH, OW, FH, FW, sh, sw);
	cout << "Q = " << Q << endl;
	
	cudaStream_t stream1; cudaStreamCreate(&stream1);
	cudaStream_t stream2; cudaStreamCreate(&stream2);

	//----------------------------------------------------------
	//winograd2D================================================
	{
		//int GN = OC;
		//int GM = N * (OH >> 1) * (OW >> 1);
		
		//__conv3D_input_pad_remode(stream1, dX, IH, IW, dD, DH, DW, N, IC, ph, pw);
		//__conv3D_winograd2D_f22x33_kernel_remode_v2(stream2, dW, dG, OC, IC);

		//wg2d_v1(NULL, 0, 0, dD, DH, DW, dG, dY, OH, OW, N, IC, OC);
		//wg2d_v2(NULL, 0, 0, dD, DH, DW, dG, dY, OH, OW, N, IC, OC);
		//wg2d_v3(NULL, 0, 0, dD, DH, DW, dG, dY, OH, OW, N, IC, OC);
		//wg2d_v4(NULL, 0, 0, dD, DH, DW, dG, dY, OH, OW, N, IC, OC);

		//conv3dWinograd2d_f22x33_k32x32R(NULL, 0, 0, dD, DH, DW, dG, dY, OH, OW, N, IC, OC);
		//conv3dWinograd2d_f22x33_k32x32R_DW4(NULL, 0, 0, dD, DH, DW, dG, dY, OH, OW, N, IC, OC);
	}

	//img2col_Winograd==========================================
	{
		__kernel_remode(NULL, dW, dCW, FH, FW, OC, IC);//[FH, FW, IC, OC]
		//__kernel_remodeV3(NULL, dW, dCW, FH, FW, OC, IC);//[FH, IC, FW, OC]
		
		//------[FH = FW = 2]------------------------------------
		{
			//===================================================
			//conv3dGemm_u88R4W2S1_ruse(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, IC, OC, ph, pw, GN, GM);
			
			//conv3dWinograd_f3x2_k64x192RC_tex(NULL, 0, texX, IH, IW, dCW, 2, dY, OH, OW, N, IC, OC, ph, pw, OW);
			//conv3dWinograd_f3x2_k64x192R6C_tex(NULL, 0, texX, IH, IW, dCW, 2, dY, OH, OW, N, IC, OC, ph, pw);
			
			//conv3dWinograd_f7x2_k64x224R_tex(NULL, texX, IH, IW, dCW, 2, dY, OH, OW, IC, OC, ph, pw);
			//conv3dWinograd_f7x2_k64x224R_p1_tex(NULL, texX, IH, IW, dCW, 2, dY, OH, OW, IC, OC, ph, pw);

			//int index = 0, GMr, GNr; 
			//conv3D_Winograd_s8_W2_64x32R_tex<2>(streams, index, length, dX, texX, IH, IW, dCW, dY, OH, OW, N, IC, OC, ph, pw, GN, GM, GMr, GNr);

			//======[FW % 2 == 0]================================
			//conv3dWinograd_SFW_f3x2_k64x192RC_tex(NULL, 0, texX, IH, IW, dCW, FH, 4, dY, OH, OW, N, IC, OC, ph, pw, OW);
		}
		
		//------[FH = FW = 3]-------------------------------------
		{
			conv3dWinogradV2_f6x3_k64x192R_p1_tex(NULL,texX, IH, IW, dCW, 3, dY, OH, OW, N, IC, OC, ph, pw);

			//===================================================
			//conv3dGemm_u88R4W3S1_ruse(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, IC, OC, ph, pw, GN, GM);
			//__conv3dWinograd_f2x3_k48R_tex<LB>(NULL, 0, 0, texX, IH, IW, dCW, dY, OH, OW, IC, OC, ph, pw, GN, GM);
			//__conv3dWinograd_f2x3_k48R_p1<LB>(stream, 0, 0, dX, IH, IW, dCW, dY, OH, OW, IC, OC, GN, GM);

			//conv3dWinograd_f2x3_k64x128R_tex(NULL, 0, 0, texX, IH, IW, dCW, 3, dY, OH, OW, IC, OC, ph, pw, GN, GM);
			//conv3dWinograd_f2x3_k64x128R4_tex(NULL, 0, 0, texX, IH, IW, dCW, 3, dY, OH, OW, IC, OC, ph, pw, GN, GM);
			
			//conv3dWinograd_f2x3_k64x128RC_tex(NULL, 0, texX, IH, IW, dCW, 3, dY, OH, OW, IC, OC, ph, pw, OW);
			//conv3dWinograd_f2x3_k64x128R4C_tex(NULL, 0, texX, IH, IW, dCW, 3, dY, OH, OW, IC, OC, ph, pw, OW);

			//conv3dWinograd_f6x3_k64x192R_tex(NULL, texX, IH, IW, dCW, 3, dY, OH, OW, N, IC, OC, ph, pw);
			//conv3dWinograd_f6x3_k64x192R_p1_tex(NULL, texX, IH, IW, dCW, 3, dY, OH, OW, N, IC, OC, ph, pw);
			//conv3dWinograd_f6x3_k64x192R_p1_OCT_tex(NULL, texX, IH, IW, dCW, 3, dY, OH, OW, N, IC, 256, ph, pw);
			//conv3dWinograd_f6x3_ruse_k64x192R_p1_tex(NULL, texX, IH, IW, dCW, 3, dY, OH, OW, N, IC, OC, ph, pw);
			
			//===================================================
			//int index = 0, GMr, GNr; 
			//conv3D_Winograd_s8_W3_64x32R_tex<3>(streams, index, length, dX, texX, IH, IW, dCW, dY, OH, OW, N, IC, OC, ph, pw, GN, GM, GMr, GNr);

			//======[FW % 3 == 0]================================
			//conv3dWinograd_SFW_f2x3_k64x128RC_tex(NULL, 0, texX, IH, IW, dCW, FH, 6, dY, OH, OW, N, IC, OC, ph, pw, OW);
			//conv3dWinograd_SFW_f2x3_k64x128R4C_tex(NULL, 0, texX, IH, IW, dCW, FH, 6, dY, OH, OW, N, IC, OC, ph, pw, OW);
		}
		
		//------[FH = FW = 4]-------------------------------------
		{
			//conv3dWinograd_f5x4_k64x160R_tex(NULL, texX, IH, IW, dCW, 4, dY, OH, OW, IC, OC, ph, pw);
			//conv3dWinograd_f5x4_k64x160R_p2_tex(NULL, texX, IH, IW, dCW, 4, dY, OH, OW, IC, OC, ph, pw);
			//conv3dWinograd_f5x4_k64x160R_p2_OCT_tex(NULL, texX, IH, IW, dCW, 4, dY, OH, OW, IC, 256, ph, pw);

			//int index = 0, GMr, GNr; 
			//conv3D_Winograd_s8_W4_64x32R_tex<4>(streams, index, length, dX, texX, IH, IW, dCW, dY, OH, OW, N, IC, OC, ph, pw, GN, GM, GMr, GNr);

			//======[FW % 4 == 0]=================================
			//conv3dWinograd_SFW_f5x4_k64x160RC_tex(NULL, 0, texX, IH, IW, dCW, FH, 8, dY, OH, OW, N, IC, OC, ph, pw, OW);
		}
		
		//------[FH = FW = 5]-------------------------------------
		{
			//conv3dWinograd_f4x5_k64x128R_tex(NULL, texX, IH, IW, dCW, 5, dY, OH, OW, N, IC, OC, ph, pw);
			//conv3dWinograd_f4x5_k64x128R_p2_tex(NULL, texX, IH, IW, dCW, 5, dY, OH, OW, N, IC, OC, ph, pw);
			//conv3dWinograd_f4x5_k64x128RC_p2_tex(NULL, 0, texX, IH, IW, dCW, 5, dY, OH, OW, N, IC, OC, ph, pw, OW);
			//conv3dWinograd_f4x5_ruse_k64x128R_p2_tex(NULL, texX, IH, IW, dCW, 5, dY, OH, OW, N, IC, OC, ph, pw);

			//int index = 0, GMr, GNr;
			//conv3D_Winograd_s8_W5_64x32R_tex<5>(streams, index, length, dX, texX, IH, IW, dCW, dY, OH, OW, N, IC, OC, ph, pw, GN, GM, GMr, GNr);
		}

		//------[FH = FW = 6]-------------------------------------
		{
			//===================================================
			//conv3dWinograd_f3x6_k64x96R_tex(NULL, texX, IH, IW, dCW, 6, dY, OH, OW, N, IC, OC, ph, pw);
			//conv3dWinograd_f3x6_k64x96R_OCT_tex(NULL, texX, IH, IW, dCW, 6, dY, OH, OW, N, IC, OC, ph, pw);
			//conv3dWinograd_f3x6_k64x96R_CT_tex(NULL, texX, IH, IW, dCW, 6, dY, OH, OW, N, IC, OC, ph, pw);
			//conv3dWinograd_f3x6_ruse_k64x96R_p3_tex(NULL, texX, IH, IW, dCW, 6, dY, OH, OW, N, IC, OC, ph, pw);

			//int index = 0, GMr, GNr;
			//conv3D_Winograd_s8_W6_64x32R_tex<6>(streams, index, length, dX, texX, IH, IW, dCW, dY, OH, OW, N, IC, OC, ph, pw, GN, GM, GMr, GNr);
		}

		//------[FH = FW = 7]-------------------------------------
		{
			//===================================================
			//conv3dGemm_u88R4W7_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);
			
			//conv3dWinograd_f2x7_k64x64R_tex(NULL, texX, IH, IW, dCW, 7, dY, OH, OW, N, IC, OC, ph, pw);
			//conv3dWinograd_f2x7_k64x64R_CT_tex(NULL, texX, IH, IW, dCW, 7, dY, OH, OW, N, 64, 64, ph, pw);

			//conv3dWinograd_f2x7_ruse_k64x64R_p3_tex(NULL, texX, IH, IW, dCW, 7, dY, OH, OW, N, IC, OC, ph, pw);
			//conv3dWinograd_f2x7_ruse_k64x64R_p3_CT_tex(NULL, texX, IH, IW, dCW, 7, dY, OH, OW, N, 64, 64, ph, pw);

			//conv3dWinograd_f2x7_k64x64RC_tex(NULL, 0, texX, IH, IW, dCW, 7, dY, OH, OW, N, IC, OC, ph, pw);
			//conv3dWinograd_f2x7_k64x64RC_CT_tex(NULL, 0, texX, IH, IW, dCW, 7, dY, OH, OW, N, 128, 128, ph, pw);
			//conv3dWinograd_f2x7_ruse_k64x64RC_p3_tex(NULL, 0, texX, IH, IW, dCW, 7, dY, OH, OW, N, IC, OC, ph, pw, OW);

			//int index = 0, GMr, GNr;
			//conv3D_Winograd_s8_W7_64x32R_tex<7>(streams, index, length, dX, texX, IH, IW, dCW, dY, OH, OW, N, IC, OC, ph, pw, GN, GM, GMr, GNr);

			//===================================================
			//conv3dWinograd_f10x7_k32x320R_p3_tex(NULL, texX, IH, IW, dCW, 7, dY, OH, OW, IC, OC, ph, pw);
			//conv3dWinograd_f10x7_k32x320R_p3_ICT_tex(NULL, texX, IH, IW, dCW, 7, dY, OH, OW, IC, 128, ph, pw);
			//conv3dWinograd_f10x7_ruse_k32x320R_p3_tex(NULL, texX, IH, IW, dCW, 7, dY, OH, OW, N, IC, OC, ph, pw);
			
			//int index = 0, GMr, GNr;
			//conv3D_Winograd_s16_W7_32x32R_p3_tex<7>(streams, index, length, dX, texX, IH, IW, dCW, dY, OH, OW, N, IC, OC, ph, pw, GN, GM, GMr, GNr);

			//===================================================
			//conv3dWinograd_f10x7_k64x320R_p3_tex(NULL, texX, IH, IW, dCW, 7, dY, OH, OW, N, IC, OC, ph, pw);

			//int index = 0, GMr, GNr;
			//conv3D_Winograd_s16_W7_64x32R_p3_tex<7>(streams, index, length, dX, texX, IH, IW, dCW, dY, OH, OW, N, IC, OC, ph, pw, GN, GM, GMr, GNr);
		}

		//------[FH = FW = 8]-------------------------------------
		{
			//===================================================
			//conv3dWinograd_f9x8_k32x288R_p4_tex(NULL, texX, IH, IW, dCW, 8, dY, OH, OW, N, IC, OC, ph, pw);

			//conv3dWinograd_f9x8_k32x288RC_p4_tex(NULL, 0, texX, IH, IW, dCW, 8, dY, OH, OW, N, IC, OC, ph, pw, OW);
			//conv3dWinograd_f9x8_k32x288RC_p4_CT_tex(NULL, 0, texX, IH, IW, dCW, 8, dY, OH, OW, N, 64, 64, ph, pw, OW);

			//conv3dWinograd_f9x8_ruse_k32x288R_p4_tex(NULL, texX, IH, IW, dCW, 8, dY, OH, OW, N, IC, OC, ph, pw);

			//int index = 0, GMr, GNr;
			//conv3D_Winograd_s16_W8_32x32R_p4_tex<8>(streams, index, length, dX, texX, IH, IW, dCW, dY, OH, OW, N, IC, OC, ph, pw, GN, GM, GMr, GNr);

			//===================================================
			//conv3dWinograd_f9x8_k64x288R_p4_tex(NULL, texX, IH, IW, dCW, 8, dY, OH, OW, N, IC, OC, ph, pw);

			//int index = 0, GMr, GNr;
			//conv3D_Winograd_s16_W8_64x32R_p4_tex<8>(streams, index, length, dX, texX, IH, IW, dCW, dY, OH, OW, N, IC, OC, ph, pw, GN, GM, GMr, GNr);
		}

		//------[FH = FW = 9]-------------------------------------
		{
			//===================================================
			//conv3dWinograd_f8x9_k32x256R_p4_tex(NULL, texX, IH, IW, dCW, 9, dY, OH, OW, N, IC, OC, ph, pw);
			//conv3dWinograd_f8x9_ruse_k32x256R_p4_tex(NULL, texX, IH, IW, dCW, 9, dY, OH, OW, N, IC, OC, ph, pw);
			
			//conv3dWinograd_f8x9_k32x256RC_p4_tex(NULL, 0, texX, IH, IW, dCW, 9, dY, OH, OW, N, IC, OC, ph, pw, OW);
			
			//int index = 0, GMr, GNr;
			//conv3D_Winograd_s16_W9_32x32R_p4_tex<9>(streams, index, length, dX, texX, IH, IW, dCW, dY, OH, OW, N, IC, OC, ph, pw, GN, GM, GMr, GNr);

			//===================================================
			//conv3dWinograd_f8x9_k64x256R_p4_tex(NULL, texX, IH, IW, dCW, 9, dY, OH, OW, N, IC, OC, ph, pw);

			//int index = 0, GMr, GNr;
			//conv3D_Winograd_s16_W9_64x32R_p4_tex<9>(streams, index, length, dX, texX, IH, IW, dCW, dY, OH, OW, N, IC, OC, ph, pw, GN, GM, GMr, GNr);
		}
	}

	//GemmR=====================================================
	{{
		//__kernel_remode(NULL, dW, dCW, FH, FW, OC, IC);

		//GemmR V2 uernel
		{
			//dr_kernel1(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, LOG2(IC), OC, sh, sw, GN, N);
			//dr_kernel2(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, LOG2(IC), OC, sh, sw, GN, N);

			//conv3dGemmV2_u88RW3p1_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, LOG2(IC), OC, sh, sw, GN, N);
			//conv3dGemmV2_u88RW4p1_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, LOG2(IC), OC, sh, sw, GN, N);
			//conv3dGemmV2_u88RW5p2_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, LOG2(IC), OC, sh, sw, GN, N);
			//conv3dGemmV2_u88R(NULL, LB, 0, 0, dX, IH, IW, dCW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, N);//passed
		}

		//GemmR V2 kernel
		{
			//conv3dGemmV2_k88RW3p1_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, LOG2(IC), OC, sh, sw, GN, N);//passed
			//conv3dGemmV2_k88RW4p1_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, LOG2(IC), OC, sh, sw, GN, N);//passed
			//conv3dGemmV2_k88RW5p2_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, LOG2(IC), OC, sh, sw, GN, N);//passed

			//conv3dGemmV2_k88R(NULL, LB, 0, 0, dX, IH, IW, dCW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, N);//passed
			//conv3dGemmV2_k84R(NULL, LB, 0, 0, dX, IH, IW, dCW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, N);//passed
			//conv3dGemmV2_k48R(NULL, LB, 0, 0, dX, IH, IW, dCW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, N);//passed
			//conv3dGemmV2_k44R(NULL, LB, 0, 0, dX, IH, IW, dCW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, N);//passed
			//conv3dGemmV2_k42R(NULL, LB, 0, 0, dX, IH, IW, dCW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, N);//passed
			//conv3dGemmV2_k24R(NULL, LB, 0, 0, dX, IH, IW, dCW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, N);//passed
		}

		//GemmR uernel reuse
		{
			//conv3dGemm_u88R4S1_ruse(NULL, LB, 0, 0, dX, IH, IW, dCW, 9, 9, dY, OH, OW, IC, OC, ph, pw, GN, GM);
			//conv3dGemm_u88R4S1_ruse(NULL, LB, 0, 0, dX, IH, IW, dCW, 7, 7, dY, OH, OW, IC, OC, ph, pw, GN, GM);

			//conv3dGemm_u88R4W3S1_ruse(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, IC, OC, ph, pw, GN, GM);
			//conv3dGemm_u88R4W2S1_ruse(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, IC, OC, ph, pw, GN, GM);
		}

		//GemmR uernel
		{
			//----------------------------------------------------------
			//conv3dGemm_u88R(NULL, LB, 0, 0, dX, IH, IW, dCW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
			//conv3dGemm_u88R_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dCW, FH, FW, dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);

			//conv3dGemm_u88R4(NULL, LB, 0, 0, dX, IH, IW, dCW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
			//conv3dGemm_u88R4_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dCW, FH, FW, dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);
			//conv3dGemm_u88R4_fw_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dCW, FH, LOG2(FW), dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);//passed

			//----------------------------------------------------------
			//conv3dGemm_u88RW3(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
			//conv3dGemm_u88RW3_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);
			
			//conv3dGemm_u88R4W3(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
			//conv3dGemm_u88R4W3_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);

			//----------------------------------------------------------
			//conv3dGemm_u88RW5(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
			//conv3dGemm_u88RW5_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);
			
			//conv3dGemm_u88R4W5(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
			//conv3dGemm_u88R4W5_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);	

			//conv3dGemm_u88R4W7_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);

			//----------------------------------------------------------
			//conv3dPure_u84R(NULL, LB, 0, 0, dX, IH, IW, dCW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
			//conv3dPure_u48R(NULL, LB, 0, 0, dX, IH, IW, dCW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
			//conv3dPure_u44R(NULL, LB, 0, 0, dX, IH, IW, dCW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
		}

		//GemmRC uernel
		{
			//conv3dGemm_u88R4CS1_ruse(NULL, LB, 0, 0, 0, dX, IH, IW, dCW, 7, 7, dY, OH, OW, IC, OC, ph, pw, GN, GM);
			//conv3dGemm_u88R4CS1_ruse(NULL, LB, 0, 0, 0, dX, IH, IW, dCW, 9, 9, dY, OH, OW, IC, OC, ph, pw, GN, GM);

			//conv3dGemm_u88RC(NULL, LB, 0, 0, 0, dX, IH, IW, dCW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
			//conv3dGemm_u88RC_ic2pow(NULL, LB, 0, 0, 0, dX, IH, IW, dCW, FH, FW, dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);
			//conv3dGemm_u88RC_fw_ic2pow(NULL, LB, 0, 0, 0, dX, IH, IW, dCW, FH, 2, dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);//FW = 4
			//conv3dGemm_u88RC_W5_ic2pow(NULL, LB, 0, 0, 0, dX, IH, IW, dCW, dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);
			//conv3dGemm_u88RC_W6_ic2pow(NULL, LB, 0, 0, 0, dX, IH, IW, dCW, dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);

			//conv3dGemm_u88R2C(NULL, LB, 0, 0, 0, dX, IH, IW, dCW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
			//conv3dGemm_u88R2C_ic2pow(NULL, LB, 0, 0, 0, dX, IH, IW, dCW, FH, FW, dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);
			//conv3dGemm_u88R2C_fw_ic2pow(NULL, LB, 0, 0, 0, dX, IH, IW, dCW, FH, 2, dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);//FW = 4
			//conv3dGemm_u88R2C_W5_ic2pow(NULL, LB, 0, 0, 0, dX, IH, IW, dCW, dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);
			//conv3dGemm_u88R2C_W6_ic2pow(NULL, LB, 0, 0, 0, dX, IH, IW, dCW, dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);
		}

		//GemmRA kernel
		{
			//conv3dGemm_k88RA(NULL, LB, 0, 0, dX, IH, IW, dCW, FH, FW, dY, OH, OW, N, IC, OC, sh, sw, ph, pw, GN, GM);//passed
			//conv3dGemm_k88RA_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dCW, FH, FW, dY, OH, OW, N, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);//passed

			//conv3dGemm_k88RA_fw_ic2pow_tex(NULL, LB, 0, 0, texX, IH, IW, dCW, FH, LOG2(FW), dY, OH, OW, N, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);//passed
			//conv3dGemm_k88RA_fw_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dCW, FH, LOG2(FW), dY, OH, OW, N, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);//passed

			//conv3dGemm_k88RA_W3(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, N, IC, OC, sh, sw, ph, pw, GN, GM);//passed
			//conv3dGemm_k88RA_W3_tex(NULL, LB, 0, 0, texX, IH, IW, dCW, dY, OH, OW, N, IC, OC, sh, sw, ph, pw, GN, GM);//passed

			//conv3dGemm_k88RA_W3_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, N, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);//passed
			//conv3dGemm_k88RA_W3_ic2pow_tex(NULL, LB, 0, 0, texX, IH, IW, dCW, dY, OH, OW, N, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);//passed

			//conv3dGemm_k88RA_W5(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, N, IC, OC, sh, sw, ph, pw, GN, GM);//passed
			//conv3dGemm_k88RA_W5_tex(NULL, LB, 0, 0, texX, IH, IW, dCW, dY, OH, OW, N, IC, OC, sh, sw, ph, pw, GN, GM);//passed
			//conv3dGemm_k88RA_W5_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, N, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);//passed
			//conv3dGemm_k88RA_W5_ic2pow_tex(NULL, LB, 0, 0, texX, IH, IW, dCW, dY, OH, OW, N, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);//passed
		}

		//FH = FW = 5
		{
			//conv3dGemm_k88RW5_tex(NULL, LB, 0, 0, texX, IH, IW, dCW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);//passed
			//conv3dGemm_k88RW5_ic2pow_tex(NULL, LB, 0, 0, texX, IH, IW, dCW, dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);//passed

			//conv3dGemm_k88R4W5_tex(NULL, LB, 0, 0, texX, IH, IW, dCW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);//passed
			//conv3dGemm_k88R4W5_ic2pow_tex(NULL, LB, 0, 0, texX, IH, IW, dCW, dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);//passed

			//conv3dGemm_k88RW5(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);//passed
			//conv3dGemm_k88RW5_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);//passed

			//conv3dGemm_k88R4W5(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);//passed
			//conv3dGemm_k88R4W5_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);//passed
		}
	
		//FH = FW = 3
		{
			//conv3dGemm_k88R4W3_ic2pow_tex(NULL, LB, 0, 0, texX, IH, IW, dCW, dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);//passed
			//conv3dGemm_k88R4W3_tex(NULL, LB, 0, 0, texX, IH, IW, dCW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);//passed
			//conv3dGemm_k88RW3_ic2pow_tex(NULL, LB, 0, 0, texX, IH, IW, dCW, dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);//passed
			//conv3dGemm_k88RW3_tex(NULL, LB, 0, 0, texX, IH, IW, dCW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);//passed

			//conv3dGemm_k88R4W3_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);//passed
			//conv3dGemm_k88RW3_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);//passed
			//conv3dGemm_k88R4W3(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);//passed
			//conv3dGemm_k88RW3(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);//passed
		}

		//GemmR 8*8 kernel
		{
			//conv3dGemm_k88R4W7_ic2pow_tex(NULL, LB, 0, 0, texX, IH, IW, dCW, dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);//passed

			//conv3dGemm_k88R4_tex(NULL, LB, 0, 0, texX, IH, IW, dCW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);//passed
			//conv3dGemm_k88R4_ic2pow_tex(NULL, LB, 0, 0, texX, IH, IW, dCW, FH, FW, dY, OH, OW, log2(IC), OC, sh, sw, ph, pw, GN, GM);//passed
			
			//conv3dGemm_k88R4_fw_ic2pow_tex(NULL, LB, 0, 0, texX, IH, IW, dCW, FH, LOG2(FW), dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);//passed
			//conv3dGemm_k88R_fw_ic2pow_tex(NULL, LB, 0, 0, texX, IH, IW, dCW, FH, LOG2(FW), dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);//passed
		
			//conv3dGemm_k88R4_fw_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dCW, FH, LOG2(FW), dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);//passed
			//conv3dGemm_k88R_fw_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dCW, FH, LOG2(FW), dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);//passed
			
			//conv3dGemm_k88R4_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dCW, FH, FW, dY, OH, OW, log2(IC), OC, sh, sw, ph, pw, GN, GM);//passed
			//conv3dGemm_k88R_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dCW, FH, FW, dY, OH, OW, log2(IC), OC, sh, sw, ph, pw, GN, GM);//passed

			//conv3dGemm_k88R4(NULL, LB, 0, 0, dX, IH, IW, dCW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);//passed
			//conv3dGemm_k88R(NULL, LB, 0, 0, dX, IH, IW, dCW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);//passed
		}
		
		//GemmR pure
		{
			//conv3dPure_k48R_tex(NULL, LB, 0, 0, texX, IH, IW, dCW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
			//conv3dPure_k28R_tex(NULL, LB, 0, 0, texX, IH, IW, dCW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);

			//conv3dPure_k84R(NULL, LB, 0, 0, dX, IH, IW, dCW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
			//conv3dPure_k48R(NULL, LB, 0, 0, dX, IH, IW, dCW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
			//conv3dPure_k44R(NULL, LB, 0, 0, dX, IH, IW, dCW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
			//conv3dPure_k82R(NULL, LB, 0, 0, dX, IH, IW, dCW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
			//conv3dPure_k42R(NULL, LB, 0, 0, dX, IH, IW, dCW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
		}

		//conv3dGemm_k84R_fw_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dCW, FH, LOG2(FW), dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);//passed
		//conv3dGemm_k44R(NULL, LB, 0, 0, dX, IH, IW, dCW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);//passed

		//GemmR W1
		{
			//conv3d_u88R_W1(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, IC, OC, GN, GM);
			//conv3d_k88R_W1(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, IC, OC, GN, GM);
			//conv3d_k84R_W1(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, IC, OC, GN, GM);
			//conv3d_k48R_W1(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, IC, OC, GN, GM);
			//conv3d_k44R_W1(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, IC, OC, GN, GM);
			//conv3d_k82R_W1(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, IC, OC, GN, GM);
			//conv3d_k42R_W1(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, IC, OC, GN, GM);
		}
	}}

	//Gemm======================================================
	{{
		//Gemm_np
		{
			//conv3dGemm_k88x4_np(NULL, LB, 0, 0, dX, IH, IW, dW, FH, FW, dY, OH, OW, IC, OC, sh, sw, GN, GM);//passed
			//conv3dGemm_k88x4_np_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dW, FH, FW, dY, OH, OW, log2(IC), OC, sh, sw, GN, GM);//passed
			//conv3dGemm_k88x4_np_fw_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dW, FH, log2(FW), dY, OH, OW, log2(IC), OC, sh, sw, GN, GM);//passed

			//conv3dGemm_k88_np(NULL, LB, 0, 0, dX, IH, IW, dW, FH, FW, dY, OH, OW, IC, OC, sh, sw, GN, GM);//passed
			//conv3dGemm_k88_np_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dW, FH, FW, dY, OH, OW, log2(IC), OC, sh, sw, GN, GM);//passed
			//conv3dGemm_k88_np_fw_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dW, FH, log2(FW), dY, OH, OW, log2(IC), OC, sh, sw, GN, GM);//passed
			//conv3dGemm_k84_np(NULL, LB, 0, 0, dX, IH, IW, dW, FH, FW, dY, OH, OW, IC, OC, sh, sw, GN, GM);//passed
			//conv3dGemm_k84_np_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dW, FH, FW, dY, OH, OW, log2(IC), OC, sh, sw, GN, GM);//passed
			//conv3dGemm_k84_np_fw_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dW, FH, log2(FW), dY, OH, OW, log2(IC), OC, sh, sw, GN, GM);//passed
			//conv3dGemm_k48_np(NULL, LB, 0, 0, dX, IH, IW, dW, FH, FW, dY, OH, OW, IC, OC, sh, sw, GN, GM);//passed
			//conv3dGemm_k48_np_fw_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dW, FH, log2(FW), dY, OH, OW, log2(IC), OC, sh, sw, GN, GM);//passed
			//conv3dGemm_k44_np_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dW, FH, FW, dY, OH, OW, log2(IC), OC, sh, sw, GN, GM);//passed
			//conv3dGemm_k44_np_fw_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dW, FH, log2(FW), dY, OH, OW, log2(IC), OC, sh, sw, GN, GM);//passed
			//conv3dGemm_k22_np(NULL, LB, 0, 0, dX, IH, IW, dW, FH, FW, dY, OH, OW, IC, OC, sh, sw, GN, GM);//passed
			//conv3dGemm_k41_np(NULL, LB, 0, 0, dX, IH, IW, dW, FH, FW, dY, OH, OW, IC, OC, sh, sw, GN, GM);//passed
			//conv3dGemm_k14_np(NULL, LB, 0, 0, dX, IH, IW, dW, FH, FW, dY, OH, OW, IC, OC, sh, sw, GN, GM);//passed
		}

		//Gemm V2
		{
			//conv3dGemmV2_k88(NULL, LB, 0, 0, dX, IH, IW, dW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, N);
			//conv3dGemmV2_k88W3p1_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dW, dY, OH, OW, LOG2(IC), OC, sh, sw, GN, N);
			//conv3dGemmV2_k88W4p1_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dW, dY, OH, OW, LOG2(IC), OC, sh, sw, GN, N);
			//conv3dGemmV2_k88W5p2_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dW, dY, OH, OW, LOG2(IC), OC, sh, sw, GN, N);
		}

		//FH = FW = 5
		{
			//conv3dGemm_k88W5(NULL, LB, 0, 0, dX, IH, IW, dW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);//passed
			//conv3dGemm_k88W5_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dW, dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);//passed
			//conv3dGemm_k88W5x4(NULL, LB, 0, 0, dX, IH, IW, dW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);//passed
			//conv3dGemm_k88W5x4_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dW, dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);//passed

			//conv3dGemm_k88W5_tex(NULL, LB, 0, 0, texX, IH, IW, dW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);//passed
			//conv3dGemm_k88W5x4_tex(NULL, LB, 0, 0, texX, IH, IW, dW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);//passed
			//conv3dGemm_k88W5_ic2pow_tex(NULL, LB, 0, 0, texX, IH, IW, dW, dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);//passed
			//conv3dGemm_k88W5x4_ic2pow_tex(NULL, LB, 0, 0, texX, IH, IW, dW, dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);//passed
		}

		//FH = FW = 3
		{
			//conv3dGemm_k88W3x4_tex(NULL, LB, 0, 0, texX, IH, IW, dW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);//passed
			//conv3dGemm_k88W3x4_ic2pow_tex(NULL, LB, 0, 0, texX, IH, IW, dW, dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);//passed
			//conv3dGemm_k88W3_tex(NULL, LB, 0, 0, texX, IH, IW, dW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);//passed
			//conv3dGemm_k88W3_ic2pow_tex(NULL, LB, 0, 0, texX, IH, IW, dW, dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);//passed

			//conv3dGemm_k88W3x4_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dW, dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);//passed
			//conv3dGemm_k88W3_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dW, dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);//passed
			//conv3dGemm_k88W3x4(NULL, LB, 0, 0, dX, IH, IW, dW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);//passed
			//conv3dGemm_k88W3(NULL, LB, 0, 0, dX, IH, IW, dW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);//passed
		}

		//Gemm 8*8 kernel
		{
			//conv3dGemm_k88x4_fw_ic2pow_tex(NULL, LB, 0, 0, texX, IH, IW, dW, FH, log2(FW), dY, OH, OW, log2(IC), OC, sh, sw, ph, pw, GN, GM);//passed
			//conv3dGemm_k88_fw_ic2pow_tex(NULL, LB, 0, 0, texX, IH, IW, dW, FH, log2(FW), dY, OH, OW, log2(IC), OC, sh, sw, ph, pw, GN, GM);//passed

			//conv3dGemm_k88x4_ic2pow_tex(NULL, LB, 0, 0, texX, IH, IW, dW, FH, FW, dY, OH, OW, log2(IC), OC, sh, sw, ph, pw, GN, GM);//passed
			//conv3dGemm_k88_ic2pow_tex(NULL, LB, 0, 0, texX, IH, IW, dW, FH, FW, dY, OH, OW, log2(IC), OC, sh, sw, ph, pw, GN, GM);//passed

			//conv3dGemm_k88x4_tex(NULL, LB, 0, 0, texX, IH, IW, dW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);//passed
			//conv3dGemm_k88_tex(NULL, LB, 0, 0, texX, IH, IW, dW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);//passed

			//====================================================================================
			//conv3dGemm_k88x4_fw_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dW, FH, LOG2(FW), dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);//passed
			//conv3dGemm_k88_fw_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dW, FH, log2(FW), dY, OH, OW, log2(IC), OC, sh, sw, ph, pw, GN, GM);//passed

			//conv3dGemm_k88x4_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dW, FH, FW, dY, OH, OW, log2(IC), OC, sh, sw, ph, pw, GN, GM);//passed
			//conv3dGemm_k88_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dW, FH, FW, dY, OH, OW, log2(IC), OC, sh, sw, ph, pw, GN, GM);//passed

			//conv3dGemm_k88x4(NULL, LB, 0, 0, dX, IH, IW, dW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);//passed
			//conv3dGemm_k88(NULL, LB, 0, 0, dX, IH, IW, dW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);//passed
		}

		//Gemm pure
		{
			//conv3dPure_k48_tex(NULL, LB, 0, 0, texX, IH, IW, dW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);

			//conv3dPure_k84(NULL, LB, 0, 0, dX, IH, IW, dW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
			//conv3dPure_k48(NULL, LB, 0, 0, dX, IH, IW, dW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
			//conv3dPure_k44(NULL, LB, 0, 0, dX, IH, IW, dW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
			//conv3dPure_k82(NULL, LB, 0, 0, dX, IH, IW, dW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
			//conv3dPure_k28(NULL, LB, 0, 0, dX, IH, IW, dW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
			//conv3dPure_k42(NULL, LB, 0, 0, dX, IH, IW, dW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
			//conv3dPure_k24(NULL, LB, 0, 0, dX, IH, IW, dW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
		}

		//conv3dGemm_k44(NULL, LB, 0, 0, dX, IH, IW, dW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);//passed
		//conv3dGemm_k42(NULL, LB, 0, 0, dX, IH, IW, dW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);//passed
		//conv3dGemm_k24(NULL, LB, 0, 0, dX, IH, IW, dW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);//passed
		//conv3dGemm_k22(NULL, LB, 0, 0, dX, IH, IW, dW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);//passed
		//conv3dGemm_k41(NULL, LB, 0, 0, dX, IH, IW, dW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);//passed
		//conv3dGemm_k21(NULL, LB, 0, 0, dX, IH, IW, dW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);//passed
		//conv3dGemm_k14(NULL, LB, 0, 0, dX, IH, IW, dW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);//passed
		//conv3dGemm_k12(NULL, LB, 0, 0, dX, IH, IW, dW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);//passed
		//conv3dGemm_k11(NULL, LB, 0, 0, dX, IH, IW, dW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);//passed
		//conv3dGemm_s1_4x2(NULL, LB, 0, 0, dX, IH, IW, dW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);//passed
		//conv3dGemm_s1_2x2(NULL, LB, 0, 0, dX, IH, IW, dW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);//passed

		//Gemm W1
		{
			//int index = 0; Conv3D(streams, index, 8, dX, IH, IW, dW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw);
			//conv3d_k88_W1(NULL, LB, 0, 0, dX, IH, IW, dW, dY, IC, OC, GN, GM);
			//conv3d_k84_W1(NULL, LB, 0, 0, dX, IH, IW, dW, dY, IC, OC, GN, GM);
			//conv3d_k48_W1(NULL, LB, 0, 0, dX, IH, IW, dW, dY, IC, OC, GN, GM);
			//conv3d_k44_W1(NULL, LB, 0, 0, dX, IH, IW, dW, dY, IC, OC, GN, GM);
			//conv3d_k82_W1(NULL, LB, 0, 0, dX, IH, IW, dW, dY, IC, OC, GN, GM);//passed
			//conv3d_k28_W1(NULL, LB, 0, 0, dX, IH, IW, dW, dY, IC, OC, GN, GM);//passed
			//conv3d_k42_W1(NULL, LB, 0, 0, dX, IH, IW, dW, dY, IC, OC, GN, GM);
			//conv3d_k24_W1(NULL, LB, 0, 0, dX, IH, IW, dW, dY, IC, OC, GN, GM);

			//conv3d_k22_W1(NULL, LB, 0, 0, dX, IH, IW, dW, dY, IC, OC, GN, GM);
			//conv3d_k21_W1(NULL, LB, 0, 0, dX, IH, IW, dW, dY, IC, OC, GN, GM);

			//conv3d_k14_W1(NULL, LB, 0, 0, dX, IH, IW, dW, dY, IC, OC, GN, GM);
			//conv3d_k12_W1(NULL, LB, 0, 0, dX, IH, IW, dW, dY, IC, OC, GN, GM);
			//conv3d_s1_4x2_W1(NULL, LB, 0, 0, dX, IH, IW, dW, dY, IC, OC, GN, GM);

			//conv3d_k11_W1(NULL, LB, 0, 0, dX, IH, IW, dW, dY, IC, OC, GN, GM);

			//int index = 0; Conv3D_W1(streams, index, 8, dX, IH, IW, dW, dY, N, IC, OC);
		}

		}}
	//--------------------------------
	error = cudaGetLastError(); printError(error);
	error = cudaMemcpy(Y3, dY, sizeof(float)*sizeY, cudaMemcpyDeviceToHost); printError(error);
	printf("GPU : "); println(Y3, 10);

	//compare--------------------------
	//float sp0 = samePercent(Y1, Y2, sizeY); cout << "sp0: " << sp0 << endl;
	
	float dif = relative_difference(Y1, Y3, sizeY); cout << "relative_difference = " << dif << endl;
	//relative_dif_dis(Y1, Y3, sizeY);

	float sp1 = samePercent(Y1, Y3, sizeY); cout << "sp1: " << sp1 << endl;
	//float sp1 = samePercent4D(Y1, Y3, N, IH, IW, IC);  cout << "sp1: " << sp1 << endl;

	//float sp2 = samePercent(Y2, Y3, sizeY); cout << "sp2: " << sp2 << endl;
	
	float zp1 = zeroPercent(Y1, sizeY); cout << "zpY1:" << zp1 << endl;
	float zp3 = zeroPercent(Y3, sizeY); cout << "zpY3:" << zp3 << endl;

	//check_zero(Y3, N, OH, OW, OC);

	//clear mem------------------------
	error = cudaFree(dX); printError(error);
	error = cudaFree(dW); printError(error);
	error = cudaFree(dY); printError(error);

	if (sp1 < 0.99f || zp1 != zp3) exit(-2);

	delete X;
	delete W;
	delete Y1; delete Y2; delete Y3;
	for (int i = 0; i < 8; i++) {
		cudaStream_t stream = (cudaStream_t)(intptr_t)streams[i];
		cudaStreamDestroy(stream);
	}
	delete[] streams;
}

template<int LB>
void testSpeed(
	int nIter,
	int IH, int IW,
	int OH, int OW,
	int FH, int FW,
	int N, int IC, int OC,
	int sh, int sw, int ph, int pw)
{
	if (OH == -1) OH = (IH + 2 * ph - FH) / sh + 1;
	if (OW == -1) OW = (IW + 2 * pw - FW) / sw + 1;

	printf("Test Speed:\n");
	printf("\t(IH, IW) = (%d, %d)\n", IH, IW);
	printf("\t(FH, FW) = (%d, %d)\n", FH, FW);
	printf("\t(n, IC, OC) = (%d, %d, %d)\n", N, IC, OC);
	printf("\t(OH, OW) = (%d, %d)\n", OH, OW);
	printf("\t(sh, sw, ph, pw) = (%d, %d, %d, %d)\n", sh, sw, ph, pw);
	const int GN = OC;
	const int GM = N * OH*OW;
	const int GK = IC * FH*FW;
	printf("\t(GN, GM, GK) = (%d, %d, %d)\n", GN, GM, GK);

	size_t sizeX = N * IC * IH * IW;
	size_t sizeW = OC * IC * FH * FW;
	size_t sizeY = N * OC * OH * OW;

	float *X = newRandomFloatVec(sizeX);
	float *W = newRandomFloatVec(sizeW);

	float *dX = newDevFloatVec(X, sizeX);
	float *dW = newDevFloatVec(W, sizeW);
	float *dY = newDevFloatVec(sizeY);
	cudaError_t error;

	float *dCW = newDevFloatVec(sizeW);
	float* dG = newDevFloatVec(OC * IC * 4 * 4);

	int DH = 0, DW = 0; float* dD; int sizeD;
	if (FH == 3 && FW == 3) {
		DH = OH - 1 + FH;
		DW = OW - 1 + FW;
		sizeD = IC * N * DH * DW;
		dD = newDevFloatVec(sizeD);
		printf("Winograd_F(2*2, 3*3): DH, DW = (%d, %d)\n", DH, DW);
	}

	if (FH == 2 && FW == 2) {
		DH = (OH + 2) / 3 ;
		DW = (OW + 2) / 3;
	}

	int length = 8;
	jlong *streams = new jlong[8];
	for (int i = 0; i < 8; i++) {
		cudaStream_t stream;
		cudaStreamCreate(&stream);
		streams[i] = (intptr_t)stream;
		//streams[i] = NULL;
	}
	
	cudaEvent_t start, end;
	cudaEventCreate(&start, cudaEventDefault);
	cudaEventCreate(&end, cudaEventDefault);

	//clock_t start = clock();
	
	cudaTextureObject_t texX = floatTexture(dX, sizeX, NULL);
	
	cudaStream_t stream1; cudaStreamCreate(&stream1);
	cudaStream_t stream2; cudaStreamCreate(&stream2);

	/*cudaStream_t stream;  cudaStreamCreate(&stream);
	cudaStreamAttrValue stream_attribute;                                                      
	stream_attribute.accessPolicyWindow.base_ptr = reinterpret_cast<void*>(dD);
	stream_attribute.accessPolicyWindow.num_bytes = sizeD << 2;

	stream_attribute.accessPolicyWindow.hitRatio = 0.9;                                        
	stream_attribute.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
	stream_attribute.accessPolicyWindow.missProp = cudaAccessPropertyPersisting;
	cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attribute);*/

	error = cudaDeviceSynchronize();
	cudaEventRecord(start, NULL);
	for (int i=0; i<nIter; i++) 
	{
		//winograd2D F(2*2, 3*3)=================================
		{
			//int GN = OC;
			//int GM = N * (OH >> 1) * (OW >> 1);
			
			//-> 6.18796
			//5.78 -> 6.40659 -> 6.96526
			//__conv3D_input_pad_remode(stream1, dX, IH, IW, dD, DH, DW, N, IC, ph, pw);
			//__conv3D_winograd2D_f22x33_kernel_remode_v2(stream2, dW, dG, OC, IC);
			
			//wg2d_v1(NULL, 0, 0, dD, DH, DW, dG, dY, OH, OW, N, IC, OC);
			//wg2d_v2(NULL, 0, 0, dD, DH, DW, dG, dY, OH, OW, N, IC, OC);
			//wg2d_v3(NULL, 0, 0, dD, DH, DW, dG, dY, OH, OW, N, IC, OC);
			//wg2d_v4(NULL, 0, 0, dD, DH, DW, dG, dY, OH, OW, N, IC, OC);

			//conv3dWinograd2d_f22x33_k32x32R(NULL, 0, 0, dD, DH, DW, dG, dY, OH, OW, N, IC, OC);
			//conv3dWinograd2d_f22x33_k32x32R_DW4(NULL, 0, 0, dD, DH, DW, dG, dY, OH, OW, N, IC, OC);
		}

		//winograd2D F(3*3, 2*2)=================================
		{
			//int GN = OC;
			//int GM = N * ((OH + 2) / 3) * ((OW + 2) / 3);

			//__conv3D_input_pad_remode(stream1, dX, IH, IW, dD, DH, DW, N, IC, ph, pw);
			//__kernel_remode(NULL, dW, dCW, FH, FW, OC, IC);
			//conv3dWinograd2d_f33x22_k32x32R(NULL, 0, 0, D, DH, DW, CW, Y, OH, OW, N, IC, OC);
		}

		//Img2col Winograd=======================================
		{
			__kernel_remode(NULL, dW, dCW, FH, FW, OC, IC);//[FH, FW, IC, OC]
			//__kernel_remodeV3(NULL, dW, dCW, FH, FW, OC, IC);//[FH, IC, FW, OC]

			//------[FH = FW = 2]-------------------------------------
			{
				//conv3dGemm_u88R4W2S1_ruse(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, IC, OC, ph, pw, GN, GM);

				//conv3dWinograd_f3x2_k64x192RC_tex(NULL, 0, texX, IH, IW, dCW, 2, dY, OH, OW, N, IC, OC, ph, pw, OW);
				//conv3dWinograd_f3x2_k64x192R6C_tex(NULL, 0, texX, IH, IW, dCW, 2, dY, OH, OW, N, IC, OC, ph, pw, OW);
				
				//conv3dWinograd_f7x2_k64x224R_tex(NULL, texX, IH, IW, dCW, 2, dY, OH, OW, IC, OC, ph, pw);
				//conv3dWinograd_f7x2_k64x224R_p1_tex(NULL, texX, IH, IW, dCW, 2, dY, OH, OW, IC, OC, ph, pw);

				//int index = 0, GMr, GNr;
				//conv3D_Winograd_s8_W2_64x32R_tex<2>(streams, index, length, dX, texX, IH, IW, dCW, dY, OH, OW, N, IC, OC, ph, pw, GN, GM, GMr, GNr);

				//FW % 2 == 0
				//conv3dWinograd_SFW_f3x2_k64x192RC_tex(NULL, 0, texX, IH, IW, dCW, FH, 4, dY, OH, OW, N, IC, OC, ph, pw, OW);
			}
			
			//------[FH = FW = 3]-------------------------------------
			{
				//conv3dWinogradV2_f6x3_k64x192R_p1_tex(NULL, texX, IH, IW, dCW, 3, dY, OH, OW, N, IC, OC, ph, pw);
			
				//===================================================================================
				//__conv3dWinograd_f2x3_k48R_tex<LB>(NULL, 0, 0, texX, IH, IW, dCW, dY, OH, OW, IC, OC, ph, pw, GN, GM);
				//__conv3dWinograd_f2x3_k48R_p1<LB>(stream, 0, 0, dX, IH, IW, dCW, dY, OH, OW, IC, OC, GN, GM);
				//conv3dGemm_u88R4W3S1(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, IC, OC, ph, pw, GN, GM);

				//conv3dWinograd_f2x3_k64x128R_tex(NULL, 0, 0, texX, IH, IW, dCW, 3, dY, OH, OW, IC, OC, ph, pw, GN, GM);
				//conv3dWinograd_f2x3_k64x128R4_tex(NULL, 0, 0, texX, IH, IW, dCW, 3, dY, OH, OW, IC, OC, ph, pw, GN, GM);
				
				//conv3dWinograd_f2x3_k64x128RC_tex(NULL, 0, texX, IH, IW, dCW, 3, dY, OH, OW, IC, OC, ph, pw, OW);
				//conv3dWinograd_f2x3_k64x128R4C_tex(NULL, 0, texX, IH, IW, dCW, 3, dY, OH, OW, IC, OC, ph, pw, OW);

				//conv3dWinograd_f6x3_k64x192R_tex(NULL, texX, IH, IW, dCW, 3, dY, OH, OW, N, IC, OC, ph, pw);
				//conv3dWinograd_f6x3_k64x192R_p1_tex(NULL, texX, IH, IW, dCW, 3, dY, OH, OW, N, IC, OC, ph, pw);
				//conv3dWinograd_f6x3_k64x192R_p1_OCT_tex(NULL, texX, IH, IW, dCW, 3, dY, OH, OW, N, IC, 128, ph, pw);

				//conv3dGemm_u88RC(NULL, 4, 0, 0, 0, dX, IH, IW, dCW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);

				//int index = 0, GMr, GNr;
				//conv3D_Winograd_s8_W3_64x32R_tex<3>(streams, index, length, dX, texX, IH, IW, dCW, dY, OH, OW, N, IC, OC, ph, pw, GN, GM, GMr, GNr);

				//------[FW % 3 == 0]-------------------------------------------
				//conv3dWinograd_SFW_f2x3_k64x128RC_tex(NULL, 0, texX, IH, IW, dCW, FH, 6, dY, OH, OW, N, IC, OC, ph, pw, OW);
				//conv3dWinograd_SFW_f2x3_k64x128R4C_tex(NULL, 0, texX, IH, IW, dCW, FH, 6, dY, OH, OW, N, IC, OC, ph, pw, OW);
			}

			//------[FH = FW = 4]-------------------------------------
			{
				//====================================================
				//conv3dWinograd_f5x4_k64x160R_tex(NULL, texX, IH, IW, dCW, 4, dY, OH, OW, IC, OC, ph, pw);
				//conv3dWinograd_f5x4_k64x160R_p2_tex(NULL, texX, IH, IW, dCW, 4, dY, OH, OW, IC, OC, ph, pw);
				//conv3dWinograd_f5x4_k64x160R_p2_OCT_tex(NULL, texX, IH, IW, dCW, 4, dY, OH, OW, IC, 256, ph, pw);

				//int index = 0, GMr, GNr;
				//conv3D_Winograd_s8_W4_64x32R_tex<4>(streams, index, length, dX, texX, IH, IW, dCW, dY, OH, OW, N, IC, OC, ph, pw, GN, GM, GMr, GNr);

				//======[FW % 8 == 0]=================================
				//conv3dWinograd_SFW_f5x4_k64x160RC_tex(NULL, 0, texX, IH, IW, dCW, FH, 8, dY, OH, OW, N, IC, OC, ph, pw, OW);
			}
			
			//------[FH = FW = 5]-------------------------------------
			{
				//conv3dWinograd_f4x5_k64x128R_tex(NULL, texX, IH, IW, dCW, 5, dY, OH, OW, N, IC, OC, ph, pw, GN, GM);
				//conv3dWinograd_f4x5_k64x128R_p2_tex(NULL, texX, IH, IW, dCW, 5, dY, OH, OW, N, IC, OC, ph, pw);
				//conv3dWinograd_f4x5_k64x128RC_p2_tex(NULL, 0, texX, IH, IW, dCW, 5, dY, OH, OW, N, IC, OC, ph, pw, OW);
				//conv3dWinograd_f4x5_ruse_k64x128R_p2_tex(NULL, texX, IH, IW, dCW, 5, dY, OH, OW, N, IC, OC, ph, pw);

				//int index = 0, GMr, GNr;
				//conv3D_Winograd_s8_W5_64x32R_tex<5>(streams, index, length, dX, texX, IH, IW, dCW, dY, OH, OW, N, IC, OC, ph, pw, GN, GM, GMr, GNr);
			}

			//------[FH = FW = 6]-------------------------------------
			{
				//conv3dWinograd_f3x6_k64x96R_tex(NULL, texX, IH, IW, dCW, 6, dY, OH, OW, N, IC, OC, ph, pw);
				//conv3dWinograd_f3x6_k64x96R_OCT_tex(NULL, texX, IH, IW, dCW, 6, dY, OH, OW, N, IC, OC, ph, pw);
				//conv3dWinograd_f3x6_k64x96R_CT_tex(NULL, texX, IH, IW, dCW, 6, dY, OH, OW, N, IC, OC, ph, pw);
				//conv3dWinograd_f3x6_ruse_k64x96R_p3_tex(NULL, texX, IH, IW, dCW, 6, dY, OH, OW, N, IC, OC, ph, pw);

				//int index = 0, GMr, GNr;
				//conv3D_Winograd_s8_W6_64x32R_tex<6>(streams, index, length, dX, texX, IH, IW, dCW, dY, OH, OW, N, IC, OC, ph, pw, GN, GM, GMr, GNr);
			}

			//------[FH = FW = 7]-------------------------------------
			{
				//conv3dGemm_u88R4W7_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);
				
				//conv3dWinograd_f2x7_k64x64R_tex(NULL, texX, IH, IW, dCW, 7, dY, OH, OW, N, IC, OC, ph, pw);
				//conv3dWinograd_f2x7_k64x64R_CT_tex(NULL, texX, IH, IW, dCW, 7, dY, OH, OW, N, 128, 128, ph, pw);

				//conv3dWinograd_f2x7_ruse_k64x64R_p3_tex(NULL, texX, IH, IW, dCW, 7, dY, OH, OW, N, IC, OC, ph, pw);
				//conv3dWinograd_f2x7_ruse_k64x64R_p3_CT_tex(NULL, texX, IH, IW, dCW, 7, dY, OH, OW, N, 64, 64, ph, pw);

				//conv3dWinograd_f2x7_k64x64RC_tex(NULL, 0, texX, IH, IW, dCW, 7, dY, OH, OW, N, IC, OC, ph, pw, OW);
				//conv3dWinograd_f2x7_k64x64RC_CT_tex(NULL, 0, texX, IH, IW, dCW, 7, dY, OH, OW, N, 256, 256, ph, pw, OW);

				//conv3dWinograd_f2x7_ruse_k64x64RC_p3_tex(NULL, 0, texX, IH, IW, dCW, 7, dY, OH, OW, N, IC, OC, ph, pw, OW);

				//int index = 0, GMr, GNr;
				//conv3D_Winograd_s8_W7_64x32R_tex<7>(streams, index, length, dX, texX, IH, IW, dCW, dY, OH, OW, N, IC, OC, ph, pw, GN, GM, GMr, GNr);
				
				//====================================================
				//conv3dWinograd_f10x7_k32x320R_p3_tex(NULL, texX, IH, IW, dCW, 7, dY, OH, OW, N, IC, OC, ph, pw);
				//conv3dWinograd_f10x7_k32x320R_p3_ICT_tex(NULL, texX, IH, IW, dCW, 7, dY, OH, OW, N, IC, OC, ph, pw);
				//conv3dWinograd_f10x7_ruse_k32x320R_p3_tex(NULL, texX, IH, IW, dCW, 7, dY, OH, OW, N, IC, OC, ph, pw);

				//int index = 0, GMr, GNr;
				//conv3D_Winograd_s16_W7_32x32R_p3_tex<7>(streams, index, length, dX, texX, IH, IW, dCW, dY, OH, OW, N, IC, OC, ph, pw, GN, GM, GMr, GNr);
				
				//====================================================
				//conv3dWinograd_f10x7_k64x320R_p3_tex(NULL, texX, IH, IW, dCW, 7, dY, OH, OW, N, IC, OC, ph, pw);

				//int index = 0, GMr, GNr;
				//conv3D_Winograd_s16_W7_64x32R_p3_tex<7>(streams, index, length, dX, texX, IH, IW, dCW, dY, OH, OW, N, IC, OC, ph, pw, GN, GM, GMr, GNr);
			}

			//------[FH = FW = 8]-------------------------------------
			{
				//conv3dWinograd_f9x8_k32x288R_p4_tex(NULL, texX, IH, IW, dCW, 8, dY, OH, OW, N, IC, OC, ph, pw);
				//conv3dWinograd_f9x8_k32x288R_p4_CT_tex(NULL, texX, IH, IW, dCW, 8, dY, OH, OW, N, 256, 256, ph, pw);

				//conv3dWinograd_f9x8_k32x288RC_p4_tex(NULL, 0, texX, IH, IW, dCW, 8, dY, OH, OW, N, IC, OC, ph, pw, OW);
				//conv3dWinograd_f9x8_k32x288RC_p4_CT_tex(NULL, 0, texX, IH, IW, dCW, 8, dY, OH, OW, N, 256, 256, ph, pw, OW);

				//conv3dWinograd_f9x8_ruse_k32x288R_p4_tex(NULL, texX, IH, IW, dCW, 8, dY, OH, OW, N, IC, OC, ph, pw);

				//int index = 0, GMr, GNr;
				//conv3D_Winograd_s16_W8_32x32R_p4_tex<8>(streams, index, length, dX, texX, IH, IW, dCW, dY, OH, OW, N, IC, OC, ph, pw, GN, GM, GMr, GNr);

				//conv3dWinograd_f9x8_k64x288R_p4_tex(NULL, texX, IH, IW, dCW, 8, dY, OH, OW, N, IC, OC, ph, pw);

				//int index = 0, GMr, GNr;
				//conv3D_Winograd_s16_W8_64x32R_p4_tex<8>(streams, index, length, dX, texX, IH, IW, dCW, dY, OH, OW, N, IC, OC, ph, pw, GN, GM, GMr, GNr);
			}

			//------[FH = FW = 9]-------------------------------------
			{
				//====================================================
				//conv3dWinograd_f8x9_k32x256R_p4_tex(NULL, texX, IH, IW, dCW, 9, dY, OH, OW, N, IC, OC, ph, pw);
				//conv3dWinograd_f8x9_k32x256R_p4_ICT_tex(NULL, texX, IH, IW, dCW, 9, dY, OH, OW, N, 128, OC, ph, pw);
				//conv3dWinograd_f8x9_k32x256R_p4_CT_tex(NULL, texX, IH, IW, dCW, 9, dY, OH, OW, N, 128, 128, ph, pw);
				
				//conv3dWinograd_f8x9_k32x256RC_p4_tex(NULL, 0, texX, IH, IW, dCW, 9, dY, OH, OW, N, IC, OC, ph, pw, OW);

				//conv3dWinograd_f8x9_ruse_k32x256R_p4_tex(NULL, texX, IH, IW, dCW, 9, dY, OH, OW, N, IC, OC, ph, pw);

				//int index = 0, GMr, GNr;
				//conv3D_Winograd_s16_W9_32x32R_p4_tex<9>(streams, index, length, dX, texX, IH, IW, dCW, dY, OH, OW, N, IC, OC, ph, pw, GN, GM, GMr, GNr);

				//====================================================
				conv3dWinograd_f8x9_k64x256R_p4_tex(NULL, texX, IH, IW, dCW, 9, dY, OH, OW, N, IC, OC, ph, pw);

				//int index = 0, GMr, GNr;
				//conv3D_Winograd_s16_W9_64x32R_p4_tex<9>(streams, index, length, dX, texX, IH, IW, dCW, dY, OH, OW, N, IC, OC, ph, pw, GN, GM, GMr, GNr);
			}
		}

		//GemmR==================================================
		{{
			//__kernel_remode(NULL, dW, dCW, FH, FW, OC, IC);

			//GemmR V2 uernel
			{
				//dr_kernel1(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, LOG2(IC), OC, sh, sw, GN, N);
				//dr_kernel2(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, LOG2(IC), OC, sh, sw, GN, N);

				//conv3dGemmV2_u88RW3p1_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, LOG2(IC), OC, sh, sw, GN, N);
				//conv3dGemmV2_u88RW4p1_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, LOG2(IC), OC, sh, sw, GN, N);
				//conv3dGemmV2_u88RW5p2_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, LOG2(IC), OC, sh, sw, GN, N);
				//conv3dGemmV2_u88R(NULL, LB, 0, 0, dX, IH, IW, dCW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, N);//passed
			}

			//GemmR V2
			{
				//conv3dGemmV2_k88RW3p1_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, LOG2(IC), OC, sh, sw, GN, N);//passed
				//conv3dGemmV2_k88RW4p1_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, LOG2(IC), OC, sh, sw, GN, N);//passed
				//conv3dGemmV2_k88RW5p2_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, LOG2(IC), OC, sh, sw, GN, N);//passed

				//conv3dGemmV2_k84R(NULL, LB, 0, 0, dX, IH, IW, dCW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, N);//passed
				//conv3dGemmV2_k48R(NULL, LB, 0, 0, dX, IH, IW, dCW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, N);//passed
				//conv3dGemmV2_k44R(NULL, LB, 0, 0, dX, IH, IW, dCW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, N);//passed
				//conv3dGemmV2_k42R(NULL, LB, 0, 0, dX, IH, IW, dCW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, N);//passed
				//conv3dGemmV2_k24R(NULL, LB, 0, 0, dX, IH, IW, dCW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, N);//passed
			}

			//GemmR uernel_ruse
			{
				//conv3dGemm_u88R4S1_ruse(NULL, LB, 0, 0, dX, IH, IW, dCW, 9, 9, dY, OH, OW, IC, OC, ph, pw, GN, GM);
				//conv3dGemm_u88R4S1_ruse(NULL, LB, 0, 0, dX, IH, IW, dCW, 7, 7, dY, OH, OW, IC, OC, ph, pw, GN, GM);

				//conv3dGemm_u88R4W3S1_ruse(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, IC, OC, ph, pw, GN, GM);
				//conv3dGemm_u88R4W2S1_ruse(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, IC, OC, ph, pw, GN, GM);
			}

			//GemmRC uernel
			{
				//conv3dGemm_u88R4CS1_ruse(NULL, LB, 0, 0, 0, dX, IH, IW, dCW, 7, 7, dY, OH, OW, IC, OC, ph, pw, GN, GM);
				//conv3dGemm_u88R4CS1_ruse(NULL, LB, 0, 0, 0, dX, IH, IW, dCW, 9, 9, dY, OH, OW, IC, OC, ph, pw, GN, GM);

				//conv3dGemm_u88RC(NULL, LB, 0, 0, 0, dX, IH, IW, dCW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
				//conv3dGemm_u88RC_ic2pow(NULL, LB, 0, 0, 0, dX, IH, IW, dCW, FH, FW, dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);
				//conv3dGemm_u88RC_fw_ic2pow(NULL, LB, 0, 0, 0, dX, IH, IW, dCW, FH, 2, dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);//FW = 4
				//conv3dGemm_u88RC_W5_ic2pow(NULL, LB, 0, 0, 0, dX, IH, IW, dCW, dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);
				//conv3dGemm_u88RC_W6_ic2pow(NULL, LB, 0, 0, 0, dX, IH, IW, dCW, dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);

				//conv3dGemm_u88R2C(NULL, LB, 0, 0, 0, dX, IH, IW, dCW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
				//conv3dGemm_u88R2C_ic2pow(NULL, LB, 0, 0, 0, dX, IH, IW, dCW, FH, FW, dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);
				//conv3dGemm_u88R2C_fw_ic2pow(NULL, LB, 0, 0, 0, dX, IH, IW, dCW, FH, 2, dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);//FW = 4
				//conv3dGemm_u88R2C_W5_ic2pow(NULL, LB, 0, 0, 0, dX, IH, IW, dCW, dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);
				//conv3dGemm_u88R2C_W6_ic2pow(NULL, LB, 0, 0, 0, dX, IH, IW, dCW, dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);
			}

			//GemmR uernel
			{
				//----------------------------------------------------------
				//conv3dGemm_u88R(NULL, LB, 0, 0, dX, IH, IW, dCW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
				//conv3dGemm_u88R_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dCW, FH, FW, dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);//passed

				//conv3dGemm_u88R4(NULL, LB, 0, 0, dX, IH, IW, dCW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
				//conv3dGemm_u88R4_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dCW, FH, FW, dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);//passed
				//conv3dGemm_u88R4_fw_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dCW, FH, LOG2(FW), dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);//passed

				//----------------------------------------------------------
				//conv3dGemm_u88RW3(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
				//conv3dGemm_u88RW3_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);

				//conv3dGemm_u88R4W3(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
				//conv3dGemm_u88R4W3_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);

				//----------------------------------------------------------
				//conv3dGemm_u88RW5(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
				//conv3dGemm_u88RW5_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);

				//conv3dGemm_u88R4W5(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
				//conv3dGemm_u88R4W5_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);

				//conv3dGemm_u88R4W7_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);

				//----------------------------------------------------------
				//conv3dPure_u84R(NULL, LB, 0, 0, dX, IH, IW, dCW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
				//conv3dPure_u48R(NULL, LB, 0, 0, dX, IH, IW, dCW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
				//conv3dPure_u44R(NULL, LB, 0, 0, dX, IH, IW, dCW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
			}

			//GemmRA
			{
				//conv3dGemm_k88RA(NULL, LB, 0, 0, dX, IH, IW, dCW, FH, FW, dY, OH, OW, N, IC, OC, sh, sw, ph, pw, GN, GM);//passed
				//conv3dGemm_k88RA_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dCW, FH, FW, dY, OH, OW, N, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);//passed

				//conv3dGemm_k88RA_fw_ic2pow_tex(NULL, LB, 0, 0, texX, IH, IW, dCW, FH, LOG2(FW), dY, OH, OW, N, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);//passed
				//conv3dGemm_k88RA_fw_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dCW, FH, LOG2(FW), dY, OH, OW, N, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);//passed

				//conv3dGemm_k88RA_W3(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, N, IC, OC, sh, sw, ph, pw, GN, GM);//passed
				//conv3dGemm_k88RA_W3_tex(NULL, LB, 0, 0, texX, IH, IW, dCW, dY, OH, OW, N, IC, OC, sh, sw, ph, pw, GN, GM);//passed

				//conv3dGemm_k88RA_W3_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, N, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);//passed
				//conv3dGemm_k88RA_W3_ic2pow_tex(NULL, LB, 0, 0, texX, IH, IW, dCW, dY, OH, OW, N, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);//passed

				//conv3dGemm_k88RA_W5(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, N, IC, OC, sh, sw, ph, pw, GN, GM);//passed
				//conv3dGemm_k88RA_W5_tex(NULL, LB, 0, 0, texX, IH, IW, dCW, dY, OH, OW, N, IC, OC, sh, sw, ph, pw, GN, GM);//passed

				//conv3dGemm_k88RA_W5_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, N, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);//passed
				//conv3dGemm_k88RA_W5_ic2pow_tex(NULL, LB, 0, 0, texX, IH, IW, dCW, dY, OH, OW, N, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);//passed
			}

			//FH = FW = 5
			{
				//conv3dGemm_k88RW5_tex(NULL, LB, 0, 0, texX, IH, IW, dCW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);//passed
				//conv3dGemm_k88RW5_ic2pow_tex(NULL, LB, 0, 0, texX, IH, IW, dCW, dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);//passed
				//conv3dGemm_k88R4W5_tex(NULL, LB, 0, 0, texX, IH, IW, dCW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);//passed
				//conv3dGemm_k88R4W5_ic2pow_tex(NULL, LB, 0, 0, texX, IH, IW, dCW, dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);//passed

				//conv3dGemm_k88RW5(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);//passed
				//conv3dGemm_k88RW5_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);//passed

				//conv3dGemm_k88R4W5(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);//passed
				//conv3dGemm_k88R4W5_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);//passed
			}

			//FH = FW = 3
			{
				//conv3dGemm_k88R4W3_ic2pow_tex(NULL, LB, 0, 0, texX, IH, IW, dCW, dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);//passed
				//conv3dGemm_k88R4W3_tex(NULL, LB, 0, 0, texX, IH, IW, dCW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);//passed

				//conv3dGemm_k88RW3_ic2pow_tex(NULL, LB, 0, 0, texX, IH, IW, dCW, dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);//passed
				//conv3dGemm_k88RW3_tex(NULL, LB, 0, 0, texX, IH, IW, dCW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);//passed

				//conv3dGemm_k88R4W3_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);//passed
				//conv3dGemm_k88RW3_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);//passed

				//conv3dGemm_k88R4W3(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);//passed
				//conv3dGemm_k88RW3(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);//passed
			}

			//GemmR 8*8 kernel
			{
				//conv3dGemm_k88R4W7_ic2pow_tex(NULL, LB, 0, 0, texX, IH, IW, dCW, dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);//passed

				//conv3dGemm_k88R4_tex(NULL, LB, 0, 0, texX, IH, IW, dCW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);//passed
				//conv3dGemm_k88R4_ic2pow_tex(NULL, LB, 0, 0, texX, IH, IW, dCW, FH, FW, dY, OH, OW, log2(IC), OC, sh, sw, ph, pw, GN, GM);//passed

				//conv3dGemm_k88R4_fw_ic2pow_tex(NULL, LB, 0, 0, texX, IH, IW, dCW, FH, LOG2(FW), dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);//passed
				//conv3dGemm_k88R_fw_ic2pow_tex(NULL, LB, 0, 0, texX, IH, IW, dCW, FH, LOG2(FW), dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);//passed

				//conv3dGemm_k88R4_fw_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dCW, FH, LOG2(FW), dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);//passed
				//conv3dGemm_k88R_fw_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dCW, FH, LOG2(FW), dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);//passed

				//conv3dGemm_k88R4_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dCW, FH, FW, dY, OH, OW, log2(IC), OC, sh, sw, ph, pw, GN, GM);//passed
				//conv3dGemm_k88R_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dCW, FH, FW, dY, OH, OW, log2(IC), OC, sh, sw, ph, pw, GN, GM);//passed

				//conv3dGemm_k88R4(NULL, LB, 0, 0, dX, IH, IW, dCW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);//passed
				//conv3dGemm_k88R(NULL, LB, 0, 0, dX, IH, IW, dCW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);//passed
			}

			//GemmR pure
			{
				//conv3dPure_k48R_tex(NULL, LB, 0, 0, texX, IH, IW, dCW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
				//conv3dPure_k28R_tex(NULL, LB, 0, 0, texX, IH, IW, dCW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);

				//conv3dPure_k84R(NULL, LB, 0, 0, dX, IH, IW, dCW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
				//conv3dPure_k48R(NULL, LB, 0, 0, dX, IH, IW, dCW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
				//conv3dPure_k44R(NULL, LB, 0, 0, dX, IH, IW, dCW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
				//conv3dPure_k82R(NULL, LB, 0, 0, dX, IH, IW, dCW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
				//conv3dPure_k42R(NULL, LB, 0, 0, dX, IH, IW, dCW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
			}

			//conv3dGemm_k84R_fw_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dCW, FH, LOG2(FW), dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);//passed
			//conv3dGemm_k44R(NULL, LB, 0, 0, dX, IH, IW, dCW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);//passed

			//GemmRW1
			{
				//conv3d_u88R_W1(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, IC, OC, GN, GM);
				//conv3d_k88R_W1(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, IC, OC, GN, GM);
				//conv3d_k84R_W1(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, IC, OC, GN, GM);
				//conv3d_k48R_W1(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, IC, OC, GN, GM);
				//conv3d_k44R_W1(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, IC, OC, GN, GM);
				//conv3d_k82R_W1(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, IC, OC, GN, GM);
				//conv3d_k42R_W1(NULL, LB, 0, 0, dX, IH, IW, dCW, dY, IC, OC, GN, GM);
			}
			}}

		//Gemm===================================================
		{{
			//Gemmnp
			{
				//conv3dGemm_k88x4_np(NULL, LB, 0, 0, dX, IH, IW, dW, FH, FW, dY, OH, OW, IC, OC, sh, sw, GN, GM);//passed
				//conv3dGemm_k88x4_np_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dW, FH, FW, dY, OH, OW, log2(IC), OC, sh, sw, GN, GM);//passed
				//conv3dGemm_k88x4_np_fw_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dW, FH, log2(FW), dY, OH, OW, log2(IC), OC, sh, sw, GN, GM);//passed

				//conv3dGemm_k88_np(NULL, LB, 0, 0, dX, IH, IW, dW, FH, FW, dY, OH, OW, IC, OC, sh, sw, GN, GM);//passed
				//conv3dGemm_k88_np_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dW, FH, FW, dY, OH, OW, log2(IC), OC, sh, sw, GN, GM);//passed
				//conv3dGemm_k88_np_fw_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dW, FH, log2(FW), dY, OH, OW, log2(IC), OC, sh, sw, GN, GM);//passed
				//conv3dGemm_k84_np(NULL, LB, 0, 0, dX, IH, IW, dW, FH, FW, dY, OH, OW, IC, OC, sh, sw, GN, GM);//passed
				//conv3dGemm_k84_np_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dW, FH, FW, dY, OH, OW, log2(IC), OC, sh, sw, GN, GM);//passed
				//conv3dGemm_k84_np_fw_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dW, FH, log2(FW), dY, OH, OW, log2(IC), OC, sh, sw, GN, GM);//passed
				//conv3dGemm_k48_np(NULL, LB, 0, 0, dX, IH, IW, dW, FH, FW, dY, OH, OW, IC, OC, sh, sw, GN, GM);//passed
				//conv3dGemm_k48_np_fw_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dW, FH, log2(FW), dY, OH, OW, log2(IC), OC, sh, sw, GN, GM);//passed
				//conv3dGemm_k44_np_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dW, FH, FW, dY, OH, OW, log2(IC), OC, sh, sw, GN, GM);//passed
				//conv3dGemm_k44_np_fw_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dW, FH, log2(FW), dY, OH, OW, log2(IC), OC, sh, sw, GN, GM);//passed
				//conv3dGemm_k22_np(NULL, LB, 0, 0, dX, IH, IW, dW, FH, FW, dY, OH, OW, IC, OC, sh, sw, GN, GM);//passed
				//conv3dGemm_k41_np(NULL, LB, 0, 0, dX, IH, IW, dW, FH, FW, dY, OH, OW, IC, OC, sh, sw, GN, GM);//passed
				//conv3dGemm_k14_np(NULL, LB, 0, 0, dX, IH, IW, dW, FH, FW, dY, OH, OW, IC, OC, sh, sw, GN, GM);//passed
			}

			//Gemm V2
			{
				//conv3dGemmV2_k88(NULL, LB, 0, 0, dX, IH, IW, dW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, N);
				//conv3dGemmV2_k88W3p1_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dW, dY, OH, OW, LOG2(IC), OC, sh, sw, GN, N);
				//conv3dGemmV2_k88W4p1_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dW, dY, OH, OW, LOG2(IC), OC, sh, sw, GN, N);
				//conv3dGemmV2_k88W5p2_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dW, dY, OH, OW, LOG2(IC), OC, sh, sw, GN, N);
			}

			//FH = FW = 5
			{
				//conv3dGemm_k88W5(NULL, LB, 0, 0, dX, IH, IW, dW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);//passed
				//conv3dGemm_k88W5_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dW, dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);//passed
				//conv3dGemm_k88W5x4(NULL, LB, 0, 0, dX, IH, IW, dW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);//passed
				//conv3dGemm_k88W5x4_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dW, dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);//passed

				//conv3dGemm_k88W5_tex(NULL, LB, 0, 0, texX, IH, IW, dW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);//passed
				//conv3dGemm_k88W5x4_tex(NULL, LB, 0, 0, texX, IH, IW, dW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);//passed
				//conv3dGemm_k88W5_ic2pow_tex(NULL, LB, 0, 0, texX, IH, IW, dW, dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);//passed
				//conv3dGemm_k88W5x4_ic2pow_tex(NULL, LB, 0, 0, texX, IH, IW, dW, dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);//passed
			}

			//FH = FW = 3
			{
				//conv3dGemm_k88W3x4_tex(NULL, LB, 0, 0, texX, IH, IW, dW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);//passed
				//conv3dGemm_k88W3x4_ic2pow_tex(NULL, LB, 0, 0, texX, IH, IW, dW, dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);//passed
				//conv3dGemm_k88W3_tex(NULL, LB, 0, 0, texX, IH, IW, dW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);//passed
				//conv3dGemm_k88W3_ic2pow_tex(NULL, LB, 0, 0, texX, IH, IW, dW, dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);//passed

				//conv3dGemm_k88W3x4_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dW, dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);//passed
				//conv3dGemm_k88W3_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dW, dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);//passed
				//conv3dGemm_k88W3x4(NULL, LB, 0, 0, dX, IH, IW, dW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);//passed
				//conv3dGemm_k88W3(NULL, LB, 0, 0, dX, IH, IW, dW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);//passed
			}

			//Gemm 8*8 kernel
			{
				//conv3dGemm_k88x4_fw_ic2pow_tex(NULL, LB, 0, 0, texX, IH, IW, dW, FH, LOG2(FW), dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);//passed
				//conv3dGemm_k88_fw_ic2pow_tex(NULL, LB, 0, 0, texX, IH, IW, dW, FH, LOG2(FW), dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);//passed

				//conv3dGemm_k88x4_ic2pow_tex(NULL, LB, 0, 0, texX, IH, IW, dW, FH, FW, dY, OH, OW, log2(IC), OC, sh, sw, ph, pw, GN, GM);//passed
				//conv3dGemm_k88_ic2pow_tex(NULL, LB, 0, 0, texX, IH, IW, dW, FH, FW, dY, OH, OW, log2(IC), OC, sh, sw, ph, pw, GN, GM);//passed

				//conv3dGemm_k88x4_tex(NULL, LB, 0, 0, texX, IH, IW, dW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);//passed
				//conv3dGemm_k88_tex(NULL, LB, 0, 0, texX, IH, IW, dW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);//passed

				//============================================================
				//conv3dGemm_k88x4_fw_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dW, FH, LOG2(FW), dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);//passed
				//conv3dGemm_k88_fw_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dW, FH, LOG2(FW), dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);//passed

				//conv3dGemm_k88x4_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dW, FH, FW, dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);//passed
				//conv3dGemm_k88_ic2pow(NULL, LB, 0, 0, dX, IH, IW, dW, FH, FW, dY, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);//passed

				//conv3dGemm_k88x4(NULL, LB, 0, 0, dX, IH, IW, dW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);//passed
				//conv3dGemm_k88(NULL, LB, 0, 0, dX, IH, IW, dW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);//passed
			}

			//Gemm pure
			{
				//conv3dPure_k48_tex(NULL, LB, 0, 0, texX, IH, IW, dW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
				//conv3dPure_k84(NULL, LB, 0, 0, dX, IH, IW, dW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
				//conv3dPure_k48(NULL, LB, 0, 0, dX, IH, IW, dW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
				//conv3dPure_k44(NULL, LB, 0, 0, dX, IH, IW, dW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
				//conv3dPure_k82(NULL, LB, 0, 0, dX, IH, IW, dW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
				//conv3dPure_k28(NULL, LB, 0, 0, dX, IH, IW, dW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
				//conv3dPure_k42(NULL, LB, 0, 0, dX, IH, IW, dW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
				//conv3dPure_k24(NULL, LB, 0, 0, dX, IH, IW, dW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
			}

			//conv3dGemm_k44(NULL, LB, 0, 0, dX, IH, IW, dW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);//passed
			//conv3dGemm_k22(NULL, LB, 0, 0, dX, IH, IW, dW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);//passed

			//conv3dGemm_k41(NULL, LB, 0, 0, dX, IH, IW, dW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);//passed
			//conv3dGemm_k21(NULL, LB, 0, 0, dX, IH, IW, dW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);//passed

			//conv3dGemm_k14(NULL, LB, 0, 0, dX, IH, IW, dW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);//passed
			//conv3dGemm_k12(NULL, LB, 0, 0, dX, IH, IW, dW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);//passed
			//conv3dGemm_k11(NULL, LB, 0, 0, dX, IH, IW, dW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);//passed
			//conv3dGemm_s1_4x2(NULL, LB, 0, 0, dX, IH, IW, dW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);//passed
			//conv3dGemm_s1_2x2(NULL, LB, 0, 0, dX, IH, IW, dW, FH, FW, dY, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);//passed

			//GemmW1
			{
				//conv3d_k88_W1(NULL, LB, 0, 0, dX, IH, IW, dW, dY, IC, OC, GN, GM);//passed
				//conv3d_k84_W1(NULL, LB, 0, 0, dX, IH, IW, dW, dY, IC, OC, GN, GM);
				//conv3d_k48_W1(NULL, LB, 0, 0, dX, IH, IW, dW, dY, IC, OC, GN, GM);
				//conv3d_k44_W1(NULL, LB, 0, 0, dX, IH, IW, dW, dY, IC, OC, GN, GM);//passed
				//conv3d_k82_W1(NULL, LB, 0, 0, dX, IH, IW, dW, dY, IC, OC, GN, GM);//passed
				//conv3d_k28_W1(NULL, LB, 0, 0, dX, IH, IW, dW, dY, IC, OC, GN, GM);//passed
				//conv3d_k42_W1(NULL, LB, 0, 0, dX, IH, IW, dW, dY,IC, OC, GN, GM);//passed
				//conv3d_k24_W1(NULL, LB, 0, 0, dX, IH, IW, dW, dY, IC, OC, GN, GM);//passed

				//conv3d_k22_W1(NULL, LB, 0, 0, dX, IH, IW, dW, dY, IC, OC, GN, GM);//passed
				//conv3d_k21_W1(NULL, LB, 0, 0, dX, IH, IW, dW, dY, IC, OC, GN, GM);

				//conv3d_k14_W1(NULL, LB, 0, 0, dX, IH, IW, dW, dY, IC, OC, GN, GM);
				//conv3d_k12_W1(NULL, LB, 0, 0, dX, IH, IW, dW, dY, IC, OC, GN, GM);
				//conv3d_s1_4x2_W1(NULL, LB, 0, 0, dX, IH, IW, dW, dY, IC, OC, GN, GM);

				//conv3d_k11_W1(NULL, LB, 0, 0, dX, IH, IW, dW, dY, IC, OC, GN, GM);

				//Conv3D_W1(pool, dX, IH, IW, dW, dY, IC, OC);
			}
		}}
		//----------------------------
		error = cudaDeviceSynchronize();
	}
	cudaDestroyTextureObject(texX);
	cudaEventRecord(end, NULL);
	
	error = cudaDeviceSynchronize(); printError(error);
	error = cudaGetLastError(); printError(error);
	//clock_t end = clock();
	
	//int div = end - start;
	float div; cudaEventElapsedTime(&div, start, end);
		
	float time = 1.0f * div / nIter;
	float size = 1.0f * GN / 1024 * GM / 1024 * GK / 1024;
	cout << "Size = " << size;
	float total = 2 * 1024 * size * 1e-9f * 1024 * 1024;
	float performance = total / (time*1e-3f);
	cout << ", Time = " << time << " msec, Performace = " << performance << " GFlop/s";

	//long long Rsize = 2 * sizeW;
	//long long Rsize = 2.78 * sizeW ;
	long long Rsize = 2 * sizeX;

	float Rspeed = (1.0f *(Rsize) / (1 << 28)) / (time*1e-3);
	cout << endl << "Rsize = " << 1.0f * Rsize / 1024 / 1024 << endl;
 	cout << endl << "Rspeed = " << Rspeed << " GB/s" << endl;

	error = cudaFree(dX); printError(error);
	error = cudaFree(dY); printError(error);
	error = cudaFree(dW); printError(error);

	delete X;
}

void test()
{
	goto STRIDE1;
	//=============================={ stride = 2 }==========================================
	{
		//int FH = 7, FW = 7, ph = 3, pw = 3, sh = 2, sw = 2;
		//int FH = 6, FW = 6, ph = 3, pw = 3, sh = 2, sw = 2;
		//int FH = 5, FW = 5, ph = 2, pw = 2, sh = 2, sw = 2;
		//int FH = 4, FW = 4, ph = 1, pw = 1, sh = 2, sw = 2;
		int FH = 3, FW = 3, ph = 1, pw = 1, sh = 2, sw = 2;

		//int IH = 112, IW = 112, N = 128, IC = 4, OC = 64;
		//int IH = 112, IW = 112, N = 128, IC = 8, OC = 64;
		//int IH = 112, IW = 112, N = 32, IC = 32, OC = 128;
		//int IH = 56, IW = 56, N = 64, IC = 128, OC = 256;
		//int IH = 28, IW = 28, OH = 14, OW = 14, N = 128, IC = 128, OC = 384;

		//int IH = 8, IW = 8, OH = 8, OW = 8, N = 64, IC = 16, OC = 64;
		//int IH = 8, IW = 8, OH = 4, OW = 4, N = 128, IC = 16, OC = 128;
		//int IH = 8, IW = 8, OH = 4, OW = 4, N = 64, IC = 16, OC = 64;

		//stride = 2:
		//int IH = 128, IW = 128, OH = 64, OW = 64, N = 128, IC = 64, OC = 128;
		int IH = 64, IW = 64, OH = 32, OW = 32, N = 128, IC = 64, OC = 128;
		//int IH = 32, IW = 32, OH = 16, OW = 16, N = 256, IC = 128, OC = 256;
		//int IH = 16, IW = 16, OH = 8, OW = 8, N = 256, IC = 256, OC = 512;
		//int IH = 8, IW = 8, OH = 4, OW = 4, N = 512, IC = 512, OC = 512;
		//int IH = 4, IW = 4, OH = 2, OW = 2, N = 512, IC = 1024, OC = 1024;


		//testCorrect<4>(IH, IW, OH, OW, FH, FW, N/4, IC, OC, sh, sw, ph, pw);
		//testCorrect<4>(IH, IW, OH, OW, FH, FW, N/4, IC/2, OC, sh, sw, ph, pw);
		//testCorrect<4>(16, 20, 8, 10, FH, FW, N/4, IC/2, OC, sh, sw, ph, pw);
		//testCorrect<4>(IH/2, IW/2, OH/2, OW/2, FH, FW, N, IC/4, OC/2, sh, sw, ph, pw);
		testSpeed<4>(1, IH, IW, OH, OW, FH, FW, N, IC, OC, sh, sw, ph, pw);
		testSpeed<4>(1000, IH, IW, OH, OW, FH, FW, N, IC, OC, sh, sw, ph, pw);
	}
	return;
	
STRIDE1:
	//=============================={ stride = 1 }==========================================
	{
		//int FH = 1, FW = 1, ph = 0, pw = 0, sh = 1, sw = 1;
		//int FH = 2, FW = 2, ph = 1, pw = 1, sh = 1, sw = 1;
		//int FH = 3, FW = 3, ph = 1, pw = 1, sh = 1, sw = 1;
		//int FH = 4, FW = 4, ph = 2, pw = 2, sh = 1, sw = 1; 
		//int FH = 5, FW = 5, ph = 2, pw = 2, sh = 1, sw = 1;
		//int FH = 6, FW = 6, ph = 3, pw = 3, sh = 1, sw = 1;
		//int FH = 7, FW = 7, ph = 3, pw = 3, sh = 1, sw = 1;
		//int FH = 8, FW = 8, ph = 4, pw = 4, sh = 1, sw = 1;
		int FH = 9, FW = 9, ph = 4, pw = 4, sh = 1, sw = 1;
	
		//int IH = 128, IW = 128, OH = 128, OW = 128, N = 32, IC = 64, OC = 64;
		//int IH = 64, IW = 64, OH = 64, OW = 64, N = 64, IC = 128, OC = 128;
		//int IH = 32, IW = 32, OH = 32, OW = 32, N = 64, IC = 256, OC = 256;

		//int IH = 32, IW = 32, OH = 32, OW = 32, N = 128, IC = 128, OC = 128;
		//int IH = 16, IW = 16, OH = 16, OW = 16, N = 128, IC = 256, OC = 256;
		//int IH = 8, IW = 8, OH = 8, OW = 8, N = 128, IC = 1024, OC = 1024;
		//int IH = 4, IW = 4, OH = 4, OW = 4, N = 128, IC = 1024, OC = 1024;

		//int IH = 224, IW = 224, OH = 224, OW = 224. N = 32, IC = 128, OC = 128;
		//int IH = 112, IW = 112, OH = 112, OW = 112, N = 32, IC = 64, OC = 64;
		//int IH = 56, IW = 56, OH = 56, OW = 56, N = 64, IC = 128, OC = 128;
		//int IH = 28, IW = 28, OH = 28, OW = 28, N = 128, IC = 128, OC = 128;
		//int IH = 14, IW = 14, OH = 14, OW = 14, N = 128, IC = 512, OC = 512;
		//int IH = 7, IW = 7, OH = 7, OW = 7, N = 128, IC = 1024, OC = 1024;

		//int IH = 96, IW = 96, OH = 96, OW = 96, N = 128, IC = 64, OC = 64;
		//int IH = 56, IW = 56, OH = 56, OW = 56, N = 128, IC = 64, OC = 64;
		//int IH = 24, IW = 24, OH = 24, OW = 24, N = 128, IC = 128, OC = 128;
		//int IH = 14, IW = 14, OH = 14, OW = 14, N = 128, IC = 256, OC = 256;
		//int IH = 6, IW = 6, OH = 6, OW = 6, N = 128, IC = 1024, OC = 1024;

		//int IH = 160, IW = 160, OH = 160, OW = 160, N = 32, IC = 128, OC = 128;
		//int IH = 120, IW = 120, OH = 120, OW = 120, N = 64, IC = 128, OC = 128;
		//int IH = 80, IW = 80, OH = 80, OW = 80, N = 64, IC = 64, OC = 64;
		//int IH = 40, IW = 40, OH = 40, OW = 40, N = 64, IC = 128, OC = 128;
		//int IH = 20, IW = 20, OH = 20, OW = 20, N = 64, IC = 256, OC = 256;
		//int IH = 10, IW = 10, OH = 10, OW = 10, N = 64, IC = 512, OC = 512;

		//======[2 * 2]========================================================
		//int IH = 112, IW = 112, OH = 112, OW = 112, N = 128, IC = 64, OC = 64;
		//int IH = 56, IW = 56, OH = 56, OW = 56, N = 128, IC = 128, OC = 128;
		//int IH = 28, IW = 28, OH = 28, OW = 28, N = 128, IC = 256, OC = 256;
		//int IH = 14, IW = 14, OH = 14, OW = 14, N = 128, IC = 512, OC = 512;

		//======[3 * 3]========================================================
		//int IH = 96, IW = 96, OH = 96, OW = 96, N = 128, IC = 64, OC = 64;
		//int IH = 48, IW = 48, OH = 48, OW = 48, N = 128, IC = 128, OC = 128;
		//int IH = 24, IW = 24, OH = 24, OW = 24, N = 128, IC = 256, OC = 256;
		//int IH = 12, IW = 12, OH = 12, OW = 12, N = 128, IC = 512, OC = 512;
		//int IH = 6, IW = 6, OH = 6, OW = 6, N = 128, IC = 1024, OC = 1024;

		//======[4 * 4]=======================================================
		//int IH = 80, IW = 80, OH = 80, OW = 80, N = 128, IC = 64, OC = 64;
		//int IH = 40, IW = 40, OH = 40, OW = 40, N = 128, IC = 128, OC = 128;
		//int IH = 20, IW = 20, OH = 20, OW = 20, N = 128, IC = 256, OC = 256;
		//int IH = 10, IW = 10, OH = 10, OW = 10, N = 128, IC = 512, OC = 512;

		//int IH = 128, IW = 128, OH = 128, OW = 128, N = 32, IC = 64, OC = 64;
		//int IH = 64, IW = 64, OH = 64, OW = 64, N = 128, IC = 64, OC = 64;
		//int IH = 16, IW = 16, OH = 16, OW = 16, N = 128, IC = 256, OC = 256;
		//int IH = 8, IW = 8, OH = 8, OW = 8, N = 128, IC = 512, OC = 512;

		//int IH = 24, IW = 24, OH = 24, OW = 24, N = 128, IC = 256, OC = 256;

		//======[5 * 5]=======================================================
		//int IH = 128, IW = 128, OH = 128, OW = 128, N = 64, IC = 64, OC = 64;
		//int IH = 64, IW = 64, OH = 64, OW = 64, N = 64, IC = 128, OC = 128;
		//int IH = 32, IW = 32, OH = 32, OW = 32, N = 128, IC = 128, OC = 128;
		//int IH = 16, IW = 16, OH = 16, OW = 16, N = 64, IC = 512, OC = 512;

		//======[6 * 6]=======================================================
		//int IH = 96, IW = 96, OH = 96, OW = 96, N = 64, IC = 64, OC = 64;
		//int IH = 48, IW = 48, OH = 48, OW = 48, N = 64, IC = 128, OC = 128;
		//int IH = 24, IW = 24, OH = 24, OW = 24, N = 64, IC = 256, OC = 256;
		//int IH = 12, IW = 12, OH = 12, OW = 12, N = 64, IC = 512, OC = 512;

		//int IH = 128, IW = 128, OH = 128, OW = 128, N = 32, IC = 64, OC = 64;
		//int IH = 32, IW = 32, OH = 32, OW = 32, N = 128, IC = 128, OC = 128;
		//int IH = 8, IW = 8, OH = 8, OW = 8, N = 128, IC = 512, OC = 512;

		//======[7 * 7]=======================================================
		//int IH = 128, IW = 128, OH = 128, OW = 128, N = 32, IC = 64, OC = 64;
		//int IH = 64, IW = 64, OH = 64, OW = 64, N = 128, IC = 64, OC = 64;
		//int IH = 32, IW = 32, OH = 32, OW = 32, N = 128, IC = 128, OC = 128;
		//int IH = 16, IW = 16, OH = 16, OW = 16, N = 128, IC = 256, OC = 256;

		//int IH = 120, IW = 120, OH = 120, OW = 120, N = 32, IC = 64, OC = 64;
		//int IH = 80, IW = 80, OH = 80, OW = 80, N = 64, IC = 64, OC = 64;
		//int IH = 40, IW = 40, OH = 40, OW = 40, N = 64, IC = 128, OC = 128;
		//int IH = 20, IW = 20, OH = 20, OW = 20, N = 64, IC = 256, OC = 256;
		//int IH = 10, IW = 10, OH = 10, OW = 10, N = 64, IC = 512, OC = 512;

		//int IH = 112, IW = 112, OH = 112, OW = 112, N = 64, IC = 64, OC = 64;

		//======[8 * 8]=======================================================
		//int IH = 72, IW = 72, OH = 72, OW = 72, N = 64, IC = 64, OC = 64;
		//int IH = 36, IW = 36, OH = 36, OW = 36, N = 64, IC = 128, OC = 128;
		//int IH = 18, IW = 18, OH = 18, OW = 18, N = 64, IC = 256, OC = 256;
		//int IH = 9, IW = 9, OH = 9, OW = 9, N = 64, IC = 512, OC = 512;

		//int IH = 128, IW = 128, OH = 128, OW = 128, N = 32, IC = 64, OC = 64;
		//int IH = 64, IW = 64, OH = 64, OW = 64, N = 128, IC = 64, OC = 64;
		//int IH = 32, IW = 32, OH = 32, OW = 32, N = 128, IC = 128, OC = 128;

		//int IH = 112, IW = 112, OH = 112, OW = 112, N = 32, IC = 64, OC = 64;
		//int IH = 56, IW = 56, OH = 56, OW = 56, N = 128, IC = 64, OC = 64;
		//int IH = 28, IW = 28, OH = 28, OW = 28, N = 128, IC = 128, OC = 128;
		//int IH = 20, IW = 20, OH = 20, OW = 20, N = 64, IC = 128, OC = 128;

		//int IH = 144, IW = 144, OH = 144, OW = 144, N = 32, IC = 64, OC = 64;
		//int IH = 72, IW = 72, OH = 72, OW = 72, N = 32, IC = 128, OC = 128;
		//int IH = 36, IW = 36, OH = 36, OW = 36, N = 32, IC = 256, OC = 256;
		//int IH = 18, IW = 18, OH = 18, OW = 18, N = 32, IC = 512, OC = 512;

		//======[9 * 9]=======================================================
		//int IH = 124, IW = 124, OH = 124, OW = 124, N = 32, IC = 64, OC = 64;
		//int IH = 60, IW = 60, OH = 60, OW = 60, N = 128, IC = 64, OC = 64;
		//int IH = 48, IW = 48, OH = 48, OW = 48, N = 128, IC = 64, OC = 64;
		//int IH = 28, IW = 28, OH = 28, OW = 28, N = 128, IC = 128, OC = 128;

		//int IH = 128, IW = 128, OH = 128, OW = 128, N = 32, IC = 64, OC = 64;
		//int IH = 64, IW = 64, OH = 64, OW = 64, N = 128, IC = 64, OC = 64;
		int IH = 32, IW = 32, OH = 32, OW = 32, N = 128, IC = 128, OC = 128;
		//int IH = 16, IW = 16, OH = 16, OW = 16, N = 128, IC = 256, OC = 256;
		//int IH = 8, IW = 8, OH = 8, OW = 8, N = 128, IC = 512, OC = 512;

		//int IH = 96, IW = 96, OH = 96, OW = 96, N = 32, IC = 64, OC = 64;

		if (FH == 2 && FW == 2) IH -= 1, IW -= 1;
		if (FH == 4 && FW == 4) IH -= 1, IW -= 1;
		if (FH == 6 && FW == 6) IH -= 1, IW -= 1;
		if (FH == 8 && FW == 8) IH -= 1, IW -= 1;

		//ph -= 1; pw -= 1; IH += 2, IW += 2;
		//ph = pw = 0; OH = OW = -1;
		//IH = IH + 2 * ph; IW = IW + 2 * pw; ph = 0; pw = 0;
		
		//IW -= 1; OW -= 1; IH -= 1; OH -= 1; 
		//IW -= 4; OW -= 4; IH -= 4; OH -= 4;
		//IH += 2; IW += 2; OH += 2; OW += 2;
		//IH += 3; IW += 3; OH += 3; OW += 3;

		//testCorrect<4>(IH, IW, OH, OW, FH, FW, N/4, IC, OC, sh, sw, ph, pw);
		//testCorrect<4>(IH, IW, OH, OW, FH, FW, N/4, IC/2, OC/2, sh, sw, ph, pw);
		//testCorrect<4>(IH/2, IW/2, OH/2, OW/2, FH, FW, N/4, IC, OC, sh, sw, ph, pw);
		//testCorrect<4>(IH/2, IW/2, OH/2, OW/2, FH, FW, N/2, IC/2, OC/2, sh, sw, ph, pw);//FH = FW = 9
		//testCorrect<4>(IH/2, IW/2, OH/2, OW/2, FH, FW, N/2, IC, OC, sh, sw, ph, pw);//FH = FW = 9

		//testCorrect<4>(IH, IW, OH, OW, FH, FW, N, IC, OC, sh, sw, ph, pw);
		testSpeed<4>(1, IH, IW, OH, OW, FH, FW, N, IC, OC, sh, sw, ph, pw);
		testSpeed<4>(1000, IH, IW, OH, OW, FH, FW, N, IC, OC, sh, sw, ph, pw);
		//testSpeed<4>(1000, IH, IW, OH, OW, FH, FW, N*4, IC, OC, sh, sw, ph, pw);
		//testSpeed<4>(1000, IH, IW, OH, OW, FH, FW, N, IC*2, OC*2, sh, sw, ph, pw);
	}
}

main() { test(); }

#endif//complie-area>>>>------------------------------------------------------------

#endif

