#pragma once

#ifndef TEST_H
#define TEST_H

#ifndef COMPLIE//<<<<complie-area--------------------------------------------------

#ifndef UTIL
#define UTIL

float* newRandomFloatVec(int length)//0-256
{
	float *p = new float[length];
	for (int i = 0; i < length; i++) p[i] = (float)(rand() % 1000) / 1000;
	return p;
}
float* newDevFloatVec(int length)
{
	float *dp = NULL;
	size_t size = sizeof(float)*length;
	cudaError_t error = cudaMalloc((void**)&dp, size); printError(error);
	error = cudaMemset(dp, 0, size); printError(error);
	return dp;
}
float* newDevFloatVec(float *p, int length)
{
	float *dp = NULL;
	size_t size = sizeof(float)*length;
	cudaError_t error = cudaMalloc((void**)&dp, size); printError(error);
	error = cudaMemcpy(dp, p, size, cudaMemcpyHostToDevice); printError(error);
	return dp;
}

char* newDevCharVec(int length)
{
	char *dp = NULL;
	size_t size = sizeof(char)*length;
	cudaError_t error = cudaMalloc((void**)&dp, size); printError(error);
	error = cudaMemset(dp, 0, size); printError(error);
	return dp;
}


void println(float *p, int length)
{
	for (int i = 0; i < length; i++) cout << p[i] << ' ';
	cout << endl;
}
float samePercent(float *a, float *b, int length)
{
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
		if (dif < 2e-4) sum++;
		//if (a[i] < b[i]) cout << i << ":" << dif << " " << a[i] << ", " << b[i] << endl;
		//else cout << i << ":" << dif << " " << a[i] << ", " << b[i] << endl;
	}
	return 1.0f*sum / length;
}

float zeroPercent(float *a, int length)
{
	int sum = 0;
	for (int i = 0; i < length; i++)
		if (a[i] == 0) sum++;
	return 1.0f*sum / length;
}

float sum(float* a, int length)
{
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


#ifndef DECONV_CPU
#define DECONV_CPU

void deconv3D_deltaX_img2col(
	float* deltaY, int OH, int OW,
	float* W, int FH, int FW,
	float* deltaX, int IH, int IW,
	int N, int IC, int OC,
	int sh, int sw, int ph, int pw)
{
	int GN = IC;
	int GM = N * IH * IW;
	int GK = OC * FH * FW;

	int OH_p = OH + (OH - 1)*(sh - 1), OW_p = OW + (OW - 1)*(sw - 1);
	int oph = FH - ph - 1, opw = FW - pw - 1;

	for (int i = 0; i < GN; i++)
	{
		int ic = i;
		for (int j = 0; j < GM; j++)
		{
			int n = j / (IH*IW);
			int j_res = j % (IH*IW);
			int ih = j_res / IW, iw = j_res % IW;

			float v = 0;
			for (int k = 0; k < GK; k++)
			{
				int oc = k / (FH*FW);
				int k_res = k % (FH*FW);
				int fh = k_res / FW, fw = k_res % FW;

				int oh = ih - oph + fh;
				int ow = iw - opw + fw;

				if (oh < 0 || ow < 0 || oh >= OH_p || ow >= OW_p) continue;
				if (oh%sh != 0 || ow % sw != 0) continue;

				v += get4d(deltaY, n, (oh / sh), (ow / sw), oc, OH, OW, OC)*
					 get4d(W, oc, (FH - 1 - fh), (FW - 1 - fw), ic, FH, FW, IC);
			}

			get4d(deltaX, n, ih, iw, ic, IH, IW, IC) = v;
		}
	}
}

void deconv3D_deltaX_img2col_s1(
	float* deltaY, int OH, int OW,
	float* W, int FH, int FW,
	float* deltaX, int IH, int IW,
	int N, int IC, int OC,
	int sh, int sw, int ph, int pw)
{
	int GN = IC;
	int GM = N * IH * IW;
	int GK = OC * FH * FW;

	int oph = FH - ph - 1;
	int opw = FW - pw - 1;

	for (int i = 0; i < GN; i++)
	{
		int ic = i;
		for (int j = 0; j < GM; j++)
		{
			int n = j / (IH*IW);
			int j_res = j % (IH*IW);
			int ih = j_res / IW, iw = j_res % IW;

			float v = 0;
			for (int k = 0; k < GK; k++)
			{
				int oc = k / (FH*FW);
				int k_res = k % (FH*FW);
				int fh = k_res / FW, fw = k_res % FW;

				int oh = ih - oph + fh;
				int ow = iw - opw + fw;

				if (oh < 0 || ow < 0 || oh >= OH || ow >= OW) continue;

				v += get4d(deltaY, n, oh, ow, oc, OH, OW, OC)*
					 get4d(W, oc, (FH - 1 - fh), (FW - 1 - fw), ic, FH, FW, IC);
			}

			get4d(deltaX, n, ih, iw, ic, IH, IW, IC) = v;
		}
	}
}
#endif


template<int LB>
void testCorrect(int IH, int IW,
	int OH, int OW,
	int FH, int FW,
	int N, int IC, int OC,
	int sh, int sw, int ph, int pw)
{
	if (IH == -1) IH = (OH - 1)*sh + FH - 2 * ph;
	if (IW == -1) IW = (OW - 1)*sw + FW - 2 * pw;

	int GN = IC;
	int GM = N * IH*IW;
	int GK = OC * FH*FW;

	printf("Test Correct:\n");
	printf("\t(OH, OW) = (%d, %d)\n", OH, OW);
	printf("\t(FH, FW) = (%d, %d)\n", FH, FW);
	printf("\t(n, IC, OC) = (%d, %d, %d)\n", N, IC, OC);
	printf("\t(IH, IW) = (%d, %d)\n", IH, IW);
	printf("\t(sh, sw, ph, pw) = (%d, %d, %d, %d)\n", sh, sw, ph, pw);
	printf("\t(GN, GM, GK) = (%d, %d, %d)\n", GN, GM, GK);

	float Qs1 = s1_PADDING_SCALE_UP(IH, IW, OH, OW, FH, FW);
	cout << "Qs1 = " << Qs1 << endl;

	float QIms2; Ims2_PADDING_SCALE_UP(QIms2, IH >> 1, IW >> 1, OH, OW, FH, FW);
	cout << "QIms2 = " << QIms2 << endl;

	int sizeX = N * IC*IH*IW;
	int sizeW = OC * IC*FH*FW;
	int sizeY = N * OC*OH*OW; cout << "sizeY = " << sizeY << endl;

	float *deltaY = newRandomFloatVec(sizeY);
	float *W = newRandomFloatVec(sizeW);

	float *deltaX1 = new float[sizeX];
	float *deltaX2 = new float[sizeX];

	//=======[winograd2D]=================================================
	int DH = IH - 1 + FH;
	int DW = IW - 1 + FW;
	float* dD = newDevFloatVec(OC * N * DH * DW);//pad: deltaY -> D
	float* dG = newDevFloatVec(OC * IC * 4 * 4);

	//=======[kernel split]===============================================
	float *dCW = 0L;
	int sizeWks = 0; {
		KS_init(N, IH, IW, FH, FW, OC, IC, sh, sw);
		sizeWks = sh * sw * OC * CFH * CFW * IC;
		dCW = newDevFloatVec(sizeWks);
	}

	//CPU-----------------------------------------------------------------
	if(sh == 1 && sw == 1) deconv3D_deltaX_img2col_s1(deltaY, OH, OW, W, FH, FW, deltaX1, IH, IW, N, IC, OC, sh, sw, ph, pw);
	else                   deconv3D_deltaX_img2col   (deltaY, OH, OW, W, FH, FW, deltaX1, IH, IW, N, IC, OC, sh, sw, ph, pw);
	cout << "CPU: "; println(deltaX1, 10);
	//float zp0 = zeroPercent(deltaX1, 10); cout << "zp0: " << zp0 << endl;

	//GPU-----------------------------------------------------------------
	float *d_deltaY = newDevFloatVec(deltaY, sizeY);
	float *dW = newDevFloatVec(W, sizeW);
	float *d_deltaX = newDevFloatVec(sizeX);

	cudaError_t error;

	int length = 8;
	jlong streams[8];
	for (int i = 0; i < length; i++) {
		cudaStream_t stream; cudaStreamCreate(&stream);
		streams[i] = (intptr_t)stream;
	}
	cudaStream_t stream; cudaStreamCreate(&stream);

	//-----------------------------------
	cudaTextureObject_t texDy = floatTexture(d_deltaY, sizeY);

	//Kernel Split Uniton=======================================
	{
		//ImsR
		{
			//Ims_init(N, IH, IW, FH, FW, OC, IC, sh, sw);
			//cout << "GN, GM = " << GN << ", " << GM << endl;
			//cout << "CFH, CFW = " << CFH << ", " << CFW << endl;

			//__ks_remodev2(NULL, dW, FH, FW, dCW, CFH, CFW, OC, IC, sh, sw);

			//float* CW = new float[sizeWks];
			//error = cudaMemcpy(CW, dCW, sizeof(float)*sizeWks, cudaMemcpyDeviceToHost); printError(error);
			//float zp_ks = zeroPercent(CW, sizeWks);
			//cout << "zp_ks = " << zp_ks << endl;

			//uernel
			{
				//ksIms_u88R8_oc2pow(NULL,  LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, LOG2(OC), sh, sw, ph, pw, GN, GM);
				//ksIms_u88R4_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, LOG2(OC), sh, sw, ph, pw, GN, GM);
			}

			//tex
			{
				//ksIms_88R8_oc2pow_tex(NULL, LB, 0, 0, texDy, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, N, IC, LOC, sh, sw, ph, pw, GN, GM);//S
				//ksIms_88R4_oc2pow_tex(NULL, LB, 0, 0, texDy, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, N, IC, LOC, sh, sw, ph, pw, GN, GM);//S
				//ksIms_88R_oc2pow_tex(NULL, LB, 0, 0, texDy, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, N, IC, LOC, sh, sw, ph, pw, GN, GM);//S

				//ksIms_88R8_tex(NULL, LB, 0, 0, texDy, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, N, IC, OC, sh, sw, ph, pw, GN, GM);//S
				//ksIms_88R4_tex(NULL, LB, 0, 0, texDy, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, N, IC, OC, sh, sw, ph, pw, GN, GM);//S
				//ksIms_88R_tex(NULL, LB, 0, 0,texDy, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, N, IC, OC, sh, sw, ph, pw, GN, GM);//S

				//ksIms_84R_tex(NULL, LB, 0, 0,texDy, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, N, IC, OC, sh, sw, ph, pw, GN, GM);//S
				//ksIms_84R_oc2pow_tex(NULL, LB, 0, 0, texDy, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, N, IC, LOC, sh, sw, ph, pw, GN, GM);//S

				//ksIms_48R_tex(NULL, LB, 0, 0,texDy, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, N, IC, OC, sh, sw, ph, pw, GN, GM);//S
				//ksIms_48R_oc2pow_tex(NULL, LB, 0, 0, texDy, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, N, IC, LOC, sh, sw, ph, pw, GN, GM);//S

				//ksIms_44R_tex(NULL, LB, 0, 0, texDy, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, N, IC, OC, sh, sw, ph, pw, GN, GM);//S
				//ksIms_44R_oc2pow_tex(NULL, LB, 0, 0, texDy, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, N, IC, LOC, sh, sw, ph, pw, GN, GM);//S
			}

			//common
			{
				//ksIms_88R8_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, N, IC, LOC, sh, sw, ph, pw, GN, GM);//S
				//ksIms_88R4_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, N, IC, LOC, sh, sw, ph, pw, GN, GM);//S
				//ksIms_88R_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, N, IC, LOC, sh, sw, ph, pw, GN, GM);//S

				//ksIms_88R8(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, N, IC, OC, sh, sw, ph, pw, GN, GM);//S
				//ksIms_88R4(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, N, IC, OC, sh, sw, ph, pw, GN, GM);//S
				//ksIms_88R(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, N, IC, OC, sh, sw, ph, pw, GN, GM);//S
		
				//ksIms_84R_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, N, IC, LOC, sh, sw, ph, pw, GN, GM);//S
				//ksIms_84R(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, N, IC, OC, sh, sw, ph, pw, GN, GM);//S

				//ksIms_48R_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, N, IC, LOC, sh, sw, ph, pw, GN, GM);//S
				//ksIms_48R(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, N, IC, OC, sh, sw, ph, pw, GN, GM);//S

				//ksIms_44R(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, N, IC, OC, sh, sw, ph, pw, GN, GM);//S
				//ksIms_44R_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, N, IC, LOC, sh, sw, ph, pw, GN, GM);//S

				//ksIms_42R(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, N, IC, OC, sh, sw, ph, pw, GN, GM);//S
				//ksIms_24R(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, N, IC, OC, sh, sw, ph, pw, GN, GM);//S
				//ksIms_22R(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, N, IC, OC, sh, sw, ph, pw, GN, GM);//S
				//ksIms_21R(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, N, IC, OC, sh, sw, ph, pw, GN, GM);//S
				//ksIms_12R(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, N, IC, OC, sh, sw, ph, pw, GN, GM);//S
				//ksIms_11R(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, N, IC, OC, sh, sw, ph, pw, GN, GM);
			}
		}

		//ImsR2
		{
			//Ims2_init(N, IH, IW, FH, FW, OC, IC);
			//cout << "GN, GM = " << GN << ", " << GM << endl;
			//cout << "CFH, CFW = " << CFH << ", " << CFW << endl;

			//__ks_remodev2(NULL, dW, FH, FW, dCW, CFH, CFW, OC, IC, sh, sw);

			//ksIms2V2
			{
				//ksV2_Ims2_u88R(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, N);
				//ksV2_Ims2_u88R_CFW_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, LOG2(OC), ph, pw, GN, N);
				//ksV2_Ims2_u88R_W5_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, d_deltaX, IH_slice, IW_slice, IC, LOG2(OC), ph, pw, GN, N);
				
				//=========================================================================================
				//ksV2_Ims2_88R_CFW_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, LOG2(OC), ph, pw, GN, N);
				//ksV2_Ims2_88R_W5_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, d_deltaX, IH_slice, IW_slice, IC, LOG2(OC), ph, pw, GN, N);

				//ksV2_Ims2_88R(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, N);
				//ksV2_Ims2_84R(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, N);
				//ksV2_Ims2_48R(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, N);
				//ksV2_Ims2_44R(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, N);
			}
			
			//uernel
			{
				//ksIms2_u88R8_oc_CFW2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, LOG2(OC), ph, pw, GN, GM);
				//ksIms2_u88R4_oc_CFW2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, LOG2(OC), ph, pw, GN, GM);
				//ksIms2_u88R_oc_CFW2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, LOG2(OC), ph, pw, GN, GM);
			}
		
			//CFW_OC_2pow
			{
				//ksIms2_88R8_oc_CFW2pow_tex(NULL, LB, 0, 0, texDy, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, LOC, ph, pw, GN, GM);
				//ksIms2_88R4_oc_CFW2pow_tex(NULL, LB, 0, 0, texDy, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, LOC, ph, pw, GN, GM);

				//ksIms2_88R8_oc_CFW2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, LOG2(OC), ph, pw, GN, GM);
				//ksIms2_88R4_oc_CFW2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, LOG2(OC), ph, pw, GN, GM);
				//ksIms2_88R_oc_CFW2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, LOG2(OC), ph, pw, GN, GM);
			}

			//OC_2POW
			{
				//ksIms2_88R8_oc2pow_tex(NULL, LB, 0, 0, texDy, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC,  LOG2(OC), ph, pw, GN, GM);
				//ksIms2_88R4_oc2pow_tex(NULL, LB, 0, 0, texDy, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, LOG2(OC), ph, pw, GN, GM);
				//ksIms2_88R_oc2pow_tex(NULL, LB, 0, 0, texDy, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, LOG2(OC), ph, pw, GN, GM);

				//ksIms2_88R8_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, LOG2(OC), ph, pw, GN, GM);
				//ksIms2_88R4_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, LOG2(OC), ph, pw, GN, GM);
				//ksIms2_88R_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, LOG2(OC), ph, pw, GN, GM);
			}
			
			//Common 8*8
			{
				//ksIms2_88R8_tex(NULL, LB, 0, 0, texDy, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, GM);
				//ksIms2_88R4_tex(NULL, LB, 0, 0, texDy, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, GM);
				//ksIms2_88R_tex(NULL, LB, 0, 0, texDy, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, GM);

				//ksIms2_88R8(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, GM);
				//ksIms2_88R4(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, GM);
				//ksIms2_88R(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, GM);
			}

			//CW 2pow
			{
				//ksIms2_88R8_CW2pow_tex(NULL, LB, 0, 0, texDy, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, GM);
				//ksIms2_88R4_CW2pow_tex(NULL, LB, 0, 0, texDy, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, GM);

				//ksIms2_88R8_CW2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, GM);
				//ksIms2_88R4_CW2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, GM);
				//ksIms2_88R_CW2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, GM);
			}

			//Kernel A
			{
				//ksIms2A_88R(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, N, IC, OC, ph, pw, GN, GM);
				//ksIms2A_88R_CW2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, N, IC, OC, ph, pw, GN, GM);
			}

			//others
			{
				//ksIms2_84R_tex(NULL, LB, 0, 0, texDy, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, GM);
				//ksIms2_84R_oc2pow_tex(NULL, LB, 0, 0, texDy, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, LOC, ph, pw, GN, GM);
				//ksIms2_48R_tex(NULL, LB, 0, 0, texDy, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, GM);
				//ksIms2_48R_oc2pow_tex(NULL, LB, 0, 0, texDy, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, LOC, ph, pw, GN, GM);
				//ksIms2_44R_tex(NULL, LB, 0, 0, texDy, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, GM);
				//ksIms2_44R_oc2pow_tex(NULL, LB, 0, 0, texDy, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, LOC, ph, pw, GN, GM);

				//====================================================================
				//ksIms2_84R(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, GM);
				//ksIms2_84R_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, LOC, ph, pw, GN, GM);

				//ksIms2_48R(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, GM);
				//ksIms2_48R_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, LOC, ph, pw, GN, GM);

				//ksIms2_44R(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, GM);
				//ksIms2_44R_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, LOC, ph, pw, GN, GM);

				//ksIms2_42R(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, GM);
				//ksIms2_24R(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, GM);
			}
		}

		//Kernel Split
		{{
			//KS_init(N, IH, IW, FH, FW, OC, IC, sh, sw);
			//cout << "GN, GM = " << GN << ", " << GM << endl;
			//cout << "CFH, CFW = " << CFH << ", " << CFW << endl;

			//int sizeWks = sh * sw * OC * CFH * CFW * IC;
			//float* dCW = newDevFloatVec(sizeWks);

			//__ks_remodev2(NULL, dW, FH, FW, dCW, CFH, CFW, OC, IC, sh, sw);
			//cudaStreamSynchronize(NULL);

			//float* CW = new float[sizeWks];
			//error = cudaMemcpy(CW, dCW, sizeof(float)*sizeWks, cudaMemcpyDeviceToHost); printError(error);
			//float zp_ks = zeroPercent(CW, sizeWks);
			//cout << "zp_ks = " << zp_ks << endl;

			//ks88R_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH, IW, N, IC, LOG2(OC), sh, sw, ph, pw, GN, GM);
			//ks84R_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH, IW, N, IC, LOG2(OC), sh, sw, ph, pw, GN, GM);
			//ks48R_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH, IW, N, IC, LOG2(OC), sh, sw, ph, pw, GN, GM);
			//ks44R_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH, IW, N, IC, LOG2(OC), sh, sw, ph, pw, GN, GM);

			//ks88R(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, GN, GM);
			//ks84R(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, GN, GM);
			//ks48R(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, GN, GM);
			//ks44R(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, GN, GM);
			//ks42R(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, GN, GM);
			//ks24R(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, GN, GM);
			//ks22R(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, GN, GM);
			//ks21R(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, GN, GM);
			//ks12R(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, GN, GM);
			//ks11R(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, GN, GM);
		}}
	}

	//Cross Add=================================================
	{{
			//float* dCW = newDevFloatVec(sizeW);
			//__kernel_remode(NULL, dW, dCW, FH, FW, OC, IC);
			//cudaMemsetAsync(d_deltaX, 0, sizeof(float)*sizeX, NULL);

			//vop_kernel1(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, OC, sh, sw, ph, pw, OC, (N*OH*OW));
			//vop_kernel2(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, OC, sh, sw, ph, pw, OC, (N*OH*OW));
			//vop_kernel3(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, OC, sh, sw, ph, pw, OC, (N*OH*OW));
			//vop_kernel4(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, OC, sh, sw, ph, pw, OC, (N*OH*OW));
			//vop_kernel5(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, OC, sh, sw, ph, pw, OC, (N*OH*OW));
			//vop_kernel6(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH, IW, IC, OC, sh, sw, ph, pw, OC, (N*OH*OW));
			//vop_kernel7(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, 3, 3, d_deltaX, IH, IW, 16, OC, sh, sw, ph, pw, OC, (N*OH*OW));
			//vop_kernel8(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, OC, sh, sw, ph, pw, OC, (N*OH*OW));
			//vop_kernel9(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, OC, sh, sw, ph, pw, OC, (N*OH*OW));
			//vop_kernel10(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, OC, sh, sw, ph, pw, OC, (N*OH*OW));
			//vop_kernel11(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, OC, sh, sw, ph, pw, OC, (N*OH*OW));

			//int index = 0; __dconv3D_deltaX_CrossAdd(streams, index, 8, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw);
			//crossAdd_k16_2(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, OC, sh, sw, ph, pw, (OC), (N*OH*OW));
			//crossAdd_k82(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, OC, sh, sw, ph, pw, (OC), (N*OH*OW));
			//crossAdd_k42(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, OC, sh, sw, ph, pw, (OC), (N*OH*OW));
			//crossAdd_k22(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, OC, sh, sw, ph, pw, (OC), (N*OH*OW));
			//crossAdd_k11(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, OC, sh, sw, ph, pw, (OC), (N*OH*OW));
	}}
	
	//Winograd2D================================================
	{
		//int GN = IC;
		//int GM = N * (IH >> 1) * (IW >> 1);

		//__deconv3D_dX_winograd2D_f22x33_kernel_remode_v2(stream, dW, dG, OC, IC);
		//__deconv3D_input_pad_remode(NULL, d_deltaY, OH, OW, dD, DH, DW, N, OC, ph, pw);
		//deconv3d_dX_Winograd2d_f22x33_k32x32R(NULL, 0, 0, dD, DH, DW, dG, d_deltaX, IH, IW, N, OC, IC, GN, GM);
	}

	//Img2col Winograd==========================================
	{
		//------[FH = FW = 2]-----------------------------------
		{
			//winograd_f3x2_k64x192C_tex(NULL, 0, texDy, OH, OW, dW, 2, d_deltaX, IH, IW, IC, OC, ph, pw, IW);
			//winograd_f3x2_k64x192x6C_tex(NULL, 0, texDy, OH, OW, dW, 2, d_deltaX, IH, IW, IC, OC, ph, pw, IW);

			//FW % 2 == 0
			//winograd_SFW_f3x2_k64x192C_tex(NULL, 0, texDy, OH, OW, dW, FH, 4, d_deltaX, IH, IW, IC, OC, ph, pw, IW);

			//============================================================================
			//winograd_f7x2_k64x224_tex(NULL, texDy, OH, OW, dW, 2, d_deltaX, IH, IW, N, IC, OC, ph, pw);
			//winograd_f7x2_k64x224_p0_tex(NULL, texDy, OH, OW, dW, 2, d_deltaX, IH, IW, N, IC, OC, ph, pw);

			/*int GNr, GMr, index = 0;
			deconv3D_dX_winograd_s8_W2_64x32R_tex<2>(streams, index, length,
				d_deltaY, texDy, OH, OW, dW, d_deltaX, IH, IW, N, IC, OC, ph, pw,
				GN, GM, GMr, GNr);*/
		}

		//------[FH = FW = 3]-----------------------------------
		{
			//winograd_f2x3_k64x128C_tex(NULL, 0, texDy, OH, OW, dW, 3, d_deltaX, IH, IW, IC, OC, ph, pw, IW);//correct
			//winograd_f2x3_k64x128x4C_tex(NULL, 0, texDy, OH, OW, dW, 3, d_deltaX, IH, IW, IC, OC, ph, pw, IW);//correct

			//winograd_f6x3_k64x192_tex(NULL, texDy, OH, OW, dW, 3, d_deltaX, IH, IW, N, IC, OC, ph, pw);//correct
			//winograd_f6x3_k64x192_p1_tex(NULL, texDy, OH, OW, dW, 3, d_deltaX, IH, IW, N, IC, OC, ph, pw);

			/*int GNr, GMr, index = 0;
			deconv3D_dX_winograd_s8_W3_64x32R_tex<3>(streams, index, length,
				d_deltaY, texDy, OH, OW, dW, d_deltaX, IH, IW, N, IC, OC, ph, pw,
				GN, GM, GMr, GNr);*/

			//FW % 3 == 0
			//winograd_SFW_f2x3_k64x128C_tex(NULL, 0, texDy, OH, OW, dW, FH, 6, d_deltaX, IH, IW, IC, OC, ph, pw, IW);
			//winograd_SFW_f2x3_k64x128x4C_tex(NULL, 0, texDy, OH, OW, dW, FH, 6, d_deltaX, IH, IW, IC, OC, ph, pw, IW);
		}

		//------[FH = FW = 4]-----------------------------------
		{
			//==================================================
			//winograd_f5x4_k64x160_tex(NULL, texDy, OH, OW, dW, 4, d_deltaX, IH, IW, N, IC, OC, ph, pw);
			//winograd_f5x4_k64x160_p1_tex(NULL, texDy, OH, OW, dW, 4, d_deltaX, IH, IW, N, IC, OC, ph, pw);

			/*int GNr, GMr, index = 0;
			deconv3D_dX_winograd_s8_W4_64x32R_tex<4>(streams, index, length,
				d_deltaY, texDy, OH, OW, dW, d_deltaX, IH, IW, N, IC, OC, ph, pw,
				GN, GM, GMr, GNr);*/

			//======[FW % 4 == 0]===============================
			//winograd_SFW_f5x4_k64x160C_tex(NULL, 0, texDy, OH, OW, dW, FH,  8, d_deltaX, IH, IW, N, IC, OC, ph, pw, IW);
		}

		//------[FH = FW = 5]-----------------------------------
		{
			//winograd_f4x5_k64x128_tex(NULL, texDy, OH, OW, dW, 5, d_deltaX, IH, IW, N, IC, OC, ph, pw);
			//winograd_f4x5_k64x128_p2_tex(NULL, texDy, OH, OW, dW, 5, d_deltaX, IH, IW, N, IC, OC, ph, pw);
			//winograd_f4x5_k64x128C_p2_tex(NULL, 0, texDy, OH, OW, dW, 5, d_deltaX, IH, IW, N, IC, OC, ph, pw, IW);
			//winograd_f4x5_ruse_k64x128_p2_tex(NULL, texDy, OH, OW, dW, 5, d_deltaX, IH, IW, N, IC, OC, ph, pw);

			/*int GNr, GMr, index = 0;
			deconv3D_dX_winograd_s8_W5_64x32R_tex<5>(streams, index, length,
				d_deltaY, texDy, OH, OW, dW, d_deltaX, IH, IW, N, IC, OC, ph, pw,
				GN, GM, GMr, GNr);*/
		}

		//------[FH = FW = 6]-----------------------------------
		{
			//winograd_f3x6_k64x96_tex(NULL, texDy, OH, OW, dW, 6, d_deltaX, IH, IW, N, IC, OC, ph, pw);
			//winograd_f3x6_k64x96_ICT_tex(NULL, texDy, OH, OW, dW, 6, d_deltaX, IH, IW, N, 256, OC, ph, pw);
			//winograd_f3x6_ruse_k64x96_p2_tex(NULL, texDy, OH, OW, dW, 6, d_deltaX, IH, IW, N, IC, OC, ph, pw);

			//winograd_f3x6_k64x96C_tex(NULL, 0, texDy, OH, OW, dW, 6, d_deltaX, IH, IW, N, IC, OC, ph, pw, IW);

			/*int GNr, GMr, index = 0;
			deconv3D_dX_winograd_s8_W6_64x32R_tex<6>(streams, index, length,
				d_deltaY, texDy, OH, OW, dW, d_deltaX, IH, IW, N, IC, OC, ph, pw,
				GN, GM, GMr, GNr);*/
		}

		//------[FH = FW = 7]-----------------------------------
		{
			//==================================================
			//winograd_f2x7_k64x64_tex(NULL, texDy, OH, OW, dW, 7, d_deltaX, IH, IW, N, IC, OC, ph, pw);
			
			//winograd_f2x7_k64x64C_tex(NULL, 0, texDy, OH, OW, dW, 7, d_deltaX, IH, IW, N, IC, OC, ph, pw, IW);
			//winograd_f2x7_k64x64C_CT_tex(NULL, 0, texDy, OH, OW, dW, 7, d_deltaX, IH, IW, N, 128, 128, ph, pw, IW);

			//winograd_f2x7_ruse_k64x64_p3_tex(NULL, texDy, OH, OW, dW, 7, d_deltaX, IH, IW, N, IC, OC, ph, pw);
			//winograd_f2x7_ruse_k64x64_p3_CT_tex(NULL, texDy, OH, OW, dW, 7, d_deltaX, IH, IW, N, 512, 512, ph, pw);

			//winograd_f2x7_ruse_k64x64C_p3_tex(NULL, 0, texDy, OH, OW, dW, 7, d_deltaX, IH, IW, N, IC, OC, ph, pw, IW);
			//winograd_f2x7_ruse_k64x64C_p3_CT_tex(NULL, 0, texDy, OH, OW, dW, 7, d_deltaX, IH, IW, N, 128, 128, ph, pw, IW);

			/*int GNr, GMr, index = 0;
			deconv3D_dX_winograd_s8_W7_64x32R_tex<7>(streams, index, length,
				d_deltaY, texDy, OH, OW, dW, d_deltaX, IH, IW, N, IC, OC, ph, pw,
				GN, GM, GMr, GNr);*/

			//==================================================
			//winograd_f10x7_k32x320_p3_tex(NULL, texDy, OH, OW, dW, 7, d_deltaX, IH, IW, N, IC, OC, ph, pw);

			/*int GNr, GMr, index = 0;
			deconv3D_dX_Winograd_s16_W7_32x32_p3_tex<7>(streams, index, length,
				d_deltaY, texDy, OH, OW, dW, d_deltaX, IH, IW, N, IC, OC, ph, pw,
				GN, GM, GMr, GNr);*/

			//==================================================
			//winograd_f10x7_k64x320_p3_tex(NULL, texDy, OH, OW, dW, 7, d_deltaX, IH, IW, N, IC, OC, ph, pw);

			/*int GNr, GMr, index = 0;
			deconv3D_dX_Winograd_s16_W7_64x32_p3_tex<7>(streams, index, length,
				d_deltaY, texDy, OH, OW, dW, d_deltaX, IH, IW, N, IC, OC, ph, pw,
				GN, GM, GMr, GNr);*/
		}

		//------[FH = FW = 8]-----------------------------------
		{
			//winograd_f9x8_k32x288_p4_tex(NULL, texDy, OH, OW, dW, 8, d_deltaX, IH, IW, N, IC, OC, ph, pw);
			//winograd_f9x8_ruse_k32x288_p4_tex(NULL, texDy, OH, OW, dW, 8, d_deltaX, IH, IW, N, IC, OC, ph, pw);

			/*int GNr, GMr, index = 0;
			deconv3D_dX_Winograd_s16_W8_32x32_p3_tex<8>(streams, index, length,
				d_deltaY, texDy, OH, OW, dW, d_deltaX, IH, IW, N, IC, OC, ph, pw,
				GN, GM, GMr, GNr);*/

			//==================================================
			//winograd_f9x8_k64x288_p4_tex(NULL, texDy, OH, OW, dW, 8, d_deltaX, IH, IW, N, IC, OC, ph, pw);

			int GNr, GMr, index = 0;
			deconv3D_dX_Winograd_s16_W8_64x32_p3_tex<8>(streams, index, length,
				d_deltaY, texDy, OH, OW, dW, d_deltaX, IH, IW, N, IC, OC, ph, pw,
				GN, GM, GMr, GNr);
		}

		//------[FH = FW = 9]-----------------------------------
		{
			//==================================================
			//winograd_f8x9_k32x256_p4_tex(NULL, texDy, OH, OW, dW, 9, d_deltaX, IH, IW, N, IC, OC, ph, pw);
			//winograd_f8x9_k32x256_p4_ICT_tex(NULL, texDy, OH, OW, dW, 9, d_deltaX, IH, IW, N, 128, OC, ph, pw);

			/*int GNr, GMr, index = 0;
			deconv3D_dX_Winograd_s16_W9_32x32_p4_tex<9>(streams, index, length,
				d_deltaY, texDy, OH, OW, dW, d_deltaX, IH, IW, N, IC, OC, ph, pw,
				GN, GM, GMr, GNr);*/

			//==================================================
			//winograd_f8x9_k64x256_p4_tex(NULL, texDy, OH, OW, dW, 9, d_deltaX, IH, IW, N, IC, OC, ph, pw);

			/*int GNr, GMr, index = 0;
			deconv3D_dX_Winograd_s16_W9_64x32_p4_tex<9>(streams, index, length,
				d_deltaY, texDy, OH, OW, dW, d_deltaX, IH, IW, N, IC, OC, ph, pw,
				GN, GM, GMr, GNr);*/
		}

		//__winograd_f2x3_k48_tex<LB>(NULL, 0, 0, texDy, OH, OW, dW, d_deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
	}

	//ZeroPadding s1============================================
	{{
		//uernel V2
		{
			//uV2_88s1(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, OC, ph, pw, GN, N);
			//uV2_88s1_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, LOG2(OC), ph, pw, GN, N);

			//uV2_88s1W3P1(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, d_deltaX, IH, IW, IC, OC, GN, N);
			//uV2_88s1W3P1_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, d_deltaX, IH, IW, IC, LOG2(OC), GN, N);

			//uV2_88s1W5P2(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, d_deltaX, IH, IW, IC, OC, GN, N);
			//uV2_88s1W5P2_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, d_deltaX, IH, IW, IC, LOG2(OC), GN, N);
		}

		//kernel V2
		{
			//kV2_88s1W5P2(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, d_deltaX, IH, IW, IC, OC, GN, N);
			//kV2_88s1W5P2_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, d_deltaX, IH, IW, IC, LOG2(OC), GN, N);

			//kV2_88s1W3P1(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, d_deltaX, IH, IW, IC, OC, GN, N);
			//kV2_88s1W3P1_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, d_deltaX, IH, IW, IC, LOG2(OC), GN, N);

			//kV2_88s1_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, LOG2(OC), ph, pw, GN, N);
			//kV2_84s1_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, LOG2(OC), ph, pw, GN, N);
			//kV2_48s1_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, LOG2(OC), ph, pw, GN, N);
			//kV2_44s1_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, LOG2(OC), ph, pw, GN, N);

			//kV2_88s1(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, OC, ph, pw, GN, N);
			//kV2_84s1(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, OC, ph, pw, GN, N);
			//kV2_48s1(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, OC, ph, pw, GN, N);
			//kV2_44s1(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, OC, ph, pw, GN, N);
		}

		//kernel A
		{
			//k88As1W3(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, d_deltaX, IH, IW, N, IC, OC, ph, pw, GN, GM);
			//k88As1W3_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, d_deltaX, IH, IW, N, IC, LOG2(OC), ph, pw, GN, GM);
			//k88As1W3_oc2pow_tex(NULL, LB, 0, 0, texDy, OH, OW, dW, d_deltaX, IH, IW, N, IC, LOG2(OC), ph, pw, GN, GM);
			
			//k88As1W5(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, d_deltaX, IH, IW, N, IC, OC, ph, pw, GN, GM);
			//k88As1W5_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, d_deltaX, IH, IW, N, IC, LOG2(OC), ph, pw, GN, GM);
			//k88As1W5_oc2pow_tex(NULL, LB, 0, 0, texDy, OH, OW, dW, d_deltaX, IH, IW, N, IC, LOG2(OC), ph, pw, GN, GM);

			//k88As1_tex(NULL, LB, 0, 0, texDy, OH, OW, dW, FH, FW, d_deltaX, IH, IW, N, IC, OC, ph, pw, GN, GM);
			//k88As1_oc2pow_tex(NULL, LB, 0, 0, texDy, OH, OW, dW, FH, FW, d_deltaX, IH, IW, N, IC, LOG2(OC), ph, pw, GN, GM);

			//k88As1(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, N, IC, OC, ph, pw, GN, GM);
			//k88As1_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, N, IC, LOG2(OC), ph, pw, GN, GM);
		}

		//uernel C
		{
			//u88s1C(NULL, LB, 0, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
			//u88s1x2C(NULL, LB, 0, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, OC, ph, pw, GN, GM);

			//u88s1x2C_oc2pow(NULL, LB, 0, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, LOG2(OC), ph, pw, GN, GM);

			//u84s1C_pure(NULL, LB, 0, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
			//u48s1C_pure(NULL, LB, 0, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
			//u44s1C_pure(NULL, LB, 0, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
		}

		//uernel ruse
		{
			//u88s1x4_ruse(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, 3, 3, d_deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
			//u88s1x4_ruse(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, 9, 9, d_deltaX, IH, IW, IC, OC, ph, pw, GN, GM);

			//u88s1W3x4_ruse(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, d_deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
		}

		//uernel
		{
			//u88s1(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
			//u88As1(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, N, IC, OC, ph, pw, GN, GM);
			//u88As1_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, N, IC, LOG2(OC), ph, pw, GN, GM);
			
			//u88As1W3(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, d_deltaX, IH, IW, N, IC, OC, ph, pw, GN, GM);
			//u88As1W3_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, d_deltaX, IH, IW, N, IC, LOG2(OC), ph, pw, GN, GM);
			//u88s1W3x4_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, d_deltaX, IH, IW, IC, LOG2(OC), ph, pw, GN, GM);

			//u88As1W5(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, d_deltaX, IH, IW, N, IC, OC, ph, pw, GN, GM);
			//u88As1W5_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, d_deltaX, IH, IW, N, IC, LOG2(OC), ph, pw, GN, GM);
			//u88s1W5x4_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, d_deltaX, IH, IW, IC, LOG2(OC), ph, pw, GN, GM);

			//u84s1_pure(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
			//u48s1_pure(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
			//u44s1_pure(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
		}

		//kernel
		{
			//f88s1(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, 3, 3, d_deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
			//f88s1(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, 5, 5, d_deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
			//f88s1x4(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, 3, 3, d_deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
			//f88s1x4(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, 5, 5, d_deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
			
			//f88s1x4_tex(NULL, LB, 0, 0, texDy, OH, OW, dW, 3, 3, d_deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
			//f88s1x4_tex(NULL, LB, 0, 0, texDy, OH, OW, dW, 5, 5, d_deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
			//f88s1_tex(NULL, LB, 0, 0, texDy, OH, OW, dW, 3, 3, d_deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
			//f88s1_tex(NULL, LB, 0, 0, texDy, OH, OW, dW, 5, 5, d_deltaX, IH, IW, IC, OC, ph, pw, GN, GM);

			//==============================================================
			//k88s1W5x4_oc2pow_tex(NULL, LB, 0, 0, texDy, OH, OW, dW, d_deltaX, IH, IW, IC, LOG2(OC), ph, pw, GN, GM);
			//k88s1W5_oc2pow_tex(NULL, LB, 0, 0, texDy, OH, OW, dW, d_deltaX, IH, IW, IC, LOG2(OC), ph, pw, GN, GM);
			
			//k88s1W5x4_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, d_deltaX, IH, IW, IC, LOG2(OC), ph, pw, GN, GM);
			//k88s1W5_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, d_deltaX, IH, IW, IC, LOG2(OC), ph, pw, GN, GM);

			//==============================================================
			//k88s1W3x4_oc2pow_tex(NULL, LB, 0, 0, texDy, OH, OW, dW, d_deltaX, IH, IW, IC, LOG2(OC), ph, pw, GN, GM);
			//k88s1W3_oc2pow_tex(NULL, LB, 0, 0, texDy, OH, OW, dW, d_deltaX, IH, IW, IC, LOG2(OC), ph, pw, GN, GM);
			
			//k88s1W3x4_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, d_deltaX, IH, IW, IC, LOG2(OC), ph, pw, GN, GM);
			//k88s1W3_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, d_deltaX, IH, IW, IC, LOG2(OC), ph, pw, GN, GM);

			//==============================================================
			//k88s1_W2pow_tex(NULL, LB, 0, 0, texDy, OH, OW, dW, LOG2(FH), LOG2(FW), d_deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
			//k88s1_tex(NULL, LB, 0, 0, texDy, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, OC, ph, pw, GN, GM);

			//k88s1_W2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, log2(FH), log2(FW), d_deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
			//k88s1(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, OC, ph, pw, GN, GM);

			//==============================================================
			//k44s1(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
			//k22s1(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
			//k21s1(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
			//k12s1(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
			//k11s1(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
		}

		//pure kernel
		{
			//k48s1_pure_tex(NULL, LB, 0, 0, texDy, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
			//k44s1_pure_tex(NULL, LB, 0, 0, texDy, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, OC, ph, pw, GN, GM);

			//k84s1_pure(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
			//k48s1_pure(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
			//k44s1_pure(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
			//k82s1_pure(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
			//k28s1_pure(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
			//k42s1_pure(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
			//k24s1_pure(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
		}
		
		//int index = 0; __dconv3D_deltaX_s1(streams, index, 8, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, N, IC, OC, ph, pw);
	}}

	//Kernel W1=================================================
	{{
		//k88W1_LB4(NULL, 0, 0, d_deltaY, dW, d_deltaX, IC, OC, GN, GM);
		//k88W1_LB3(NULL, 0, 0, d_deltaY, dW, d_deltaX, IC, OC, GN, GM);
		//k84W1(NULL, LB, 0, 0, d_deltaY, dW, d_deltaX, IC, OC, GN, GM);
		//k48W1(NULL, LB, 0, 0, d_deltaY, dW, d_deltaX, IC, OC, GN, GM);
		//k44W1(NULL, LB, 0, 0, d_deltaY, dW, d_deltaX, IC, OC, GN, GM);
		//k82W1(NULL, LB, 0, 0, d_deltaY, dW, d_deltaX, IC, OC, GN, GM);
		//k28W1(NULL, LB, 0, 0, d_deltaY, dW, d_deltaX, IC, OC, GN, GM);
		//k42W1(NULL, LB, 0, 0, d_deltaY, dW, d_deltaX, IC, OC, GN, GM);
		//k24W1(NULL, LB, 0, 0, d_deltaY, dW, d_deltaX, IC, OC, GN, GM);
		//k22W1(NULL, LB, 0, 0, d_deltaY, dW, d_deltaX, IC, OC, GN, GM);
		//k21W1(NULL, LB, 0, 0, d_deltaY, dW, d_deltaX, IC, OC, GN, GM);
		//k12W1(NULL, LB, 0, 0, d_deltaY, dW, d_deltaX, IC, OC, GN, GM);
		//k11W1(NULL, LB, 0, 0, d_deltaY, dW, d_deltaX, IC, OC, GN, GM);
		//int index = 0; __dconv3D_deltaX_W1(streams, index, 8, d_deltaY, dW, d_deltaX, IH, IW, IC, OC);
	}}

	error = cudaDeviceSynchronize(); printError(error);
	error = cudaMemcpy(deltaX2, d_deltaX, sizeof(float)*sizeX, cudaMemcpyDeviceToHost); printError(error);
	cout << "GPU: "; println(deltaX2, 10);
	float sp  = samePercent(deltaX1, deltaX2, sizeX); cout << "sp: " << sp << endl;
	float zp1 = zeroPercent(deltaX2, sizeX); cout << "zp1: " << zp1 << endl;

	error = cudaGetLastError(); printError(error);
	
	//check_zero(deltaX2, N, IH, IW, IC);

	//error = cudaFree(d_deltaY); printError(error);
	//error = cudaFree(dW); printError(error);
	//error = cudaFree(d_deltaX); printError(error);
	//error = cudaDestroyTextureObject(texDy); printError(error);

	delete deltaY;
	delete W;
	delete deltaX1;
	delete deltaX2;

	if (sp < 0.99f) {exit(2);}
}

template<int LB>
void testSpeed(int nIter,
	int IH, int IW,
	int OH, int OW,
	int FH, int FW,
	int N, int IC, int OC,
	int sh, int sw, int ph, int pw)
{
	if (IH == -1) IH = (OH - 1)*sh + FH - 2 * ph;
	if (IW == -1) IW = (OW - 1)*sw + FW - 2 * pw;

	int GN = IC;
	int GM = N * IH*IW;
	int GK = OC * FH*FW;

	printf("Test Speed:\n");
	printf("\t(OH, OW) = (%d, %d)\n", OH, OW);
	printf("\t(FH, FW) = (%d, %d)\n", FH, FW);
	printf("\t(n, IC, OC) = (%d, %d, %d)\n", N, IC, OC);
	printf("\t(IH, IW) = (%d, %d)\n", IH, IW);
	printf("\t(sh, sw, ph, pw) = (%d, %d, %d, %d)\n", sh, sw, ph, pw);
	printf("\t(GN, GM, GK) = (%d, %d, %d)\n", GN, GM, GK);

	int sizeX = N  * IC * IH * IW;
	int sizeW = OC * IC * FH * FW;
	int sizeY = N  * OC * OH * OW;

	float* deltaY = newRandomFloatVec(sizeY);
	//float* W = newRandomFloatVec(sizeW);

	float* d_deltaY = newDevFloatVec(deltaY, sizeY);
	float* dW = newDevFloatVec(sizeW);
	float* d_deltaX = newDevFloatVec(sizeX);
	cudaError_t error;

	int length = 8;
	jlong streams[8];
	for (int i = 0; i < length; i++) {
		cudaStream_t stream; cudaStreamCreate(&stream);
		streams[i] = (intptr_t)stream;
	}
	cudaStream_t stream; cudaStreamCreate(&stream);

	cudaTextureObject_t texDy = floatTexture(d_deltaY, sizeY);

	float *dCW = 0L;
	int sizeWks = 0; {
		KS_init(N, IH, IW, FH, FW, OC, IC, sh, sw);
		sizeWks = sh * sw * OC * CFH * CFW * IC;
		dCW = newDevFloatVec(sizeWks);
	}

	//winograd2D
	int DH = IH - 1 + FH;
	int DW = IW - 1 + FW;
	float* dD = newDevFloatVec(OC * N * DH * DW);//pad: deltaY -> D
	float* dG = newDevFloatVec(OC * IC * 4 * 4);

	error = cudaDeviceSynchronize(); printError(error);
	clock_t start = clock();
	for (int i = 0; i < nIter; i++)
	{
		//Kernel Split Union=====================================================
		{
			//KernelSplit ImsR===================================================
			{
				//Ims_init(N, IH, IW, FH, FW, OC, IC, sh, sw);

				//__ks_remodev2(NULL, dW, FH, FW, dCW, CFH, CFW, OC, IC, sh, sw);
				//cout << GN << ", " << GM << endl;
				//cout << (GN >> LB >> 3) << ", " << (GM >> LB >> 3) << endl;

				//uernel
				{
					//ksIms_u88R8_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, LOG2(OC), sh, sw, ph, pw, GN, GM);
					//ksIms_u88R4_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, LOG2(OC), sh, sw, ph, pw, GN, GM);
				}

				//tex
				{
					//ksIms_88R8_oc2pow_tex(NULL, LB, 0, 0, texDy, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, N, IC, LOC, sh, sw, ph, pw, GN, GM);
					//ksIms_88R4_oc2pow_tex(NULL, LB, 0, 0, texDy, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, N, IC, LOC, sh, sw, ph, pw, GN, GM);
					//ksIms_88R_oc2pow_tex(NULL, LB, 0, 0, texDy, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, N, IC, LOC, sh, sw, ph, pw, GN, GM);

					//ksIms_88R8_tex(NULL, LB, 0, 0, texDy, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, N, IC, OC, sh, sw, ph, pw, GN, GM);
					//ksIms_88R4_tex(NULL, LB, 0, 0, texDy, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, N, IC, OC, sh, sw, ph, pw, GN, GM);
					//ksIms_88R_tex(NULL, LB, 0, 0, texDy, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, N, IC, OC, sh, sw, ph, pw, GN, GM);

					//ksIms_84R_tex(NULL, LB, 0, 0, texDy, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, N, IC, OC, sh, sw, ph, pw, GN, GM);
					//ksIms_84R_oc2pow_tex(NULL, LB, 0, 0, texDy, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, N, IC, LOC, sh, sw, ph, pw, GN, GM);
	
					//ksIms_48R_tex(NULL, LB, 0, 0,texDy, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, N, IC, OC, sh, sw, ph, pw, GN, GM);
					//ksIms_48R_oc2pow_tex(NULL, LB, 0, 0, texDy, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, N, IC, LOC, sh, sw, ph, pw, GN, GM);

					//ksIms_44R_tex(NULL, LB, 0, 0, texDy, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, N, IC, OC, sh, sw, ph, pw, GN, GM);
					//ksIms_44R_oc2pow_tex(NULL, LB, 0, 0, texDy, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, N, IC, LOC, sh, sw, ph, pw, GN, GM);
				}
				
				//common
				{
					//ksIms_88R8_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, N, IC, LOC, sh, sw, ph, pw, GN, GM);
					//ksIms_88R4_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, N, IC, LOC, sh, sw, ph, pw, GN, GM);
					//ksIms_88R_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, N, IC, LOC, sh, sw, ph, pw, GN, GM);

					//ksIms_88R8(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, N, IC, OC, sh, sw, ph, pw, GN, GM);
					//ksIms_88R4(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, N, IC, OC, sh, sw, ph, pw, GN, GM);
					//ksIms_88R(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, N, IC, OC, sh, sw, ph, pw, GN, GM);

					//ksIms_84R_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, N, IC, LOC, sh, sw, ph, pw, GN, GM);
					//ksIms_84R(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, N, IC, OC, sh, sw, ph, pw, GN, GM);
	
					//ksIms_48R_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, N, IC, LOC, sh, sw, ph, pw, GN, GM);
					//ksIms_48R(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, N, IC, OC, sh, sw, ph, pw, GN, GM);

					//ksIms_44R(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, N, IC, OC, sh, sw, ph, pw, GN, GM);
					//ksIms_44R_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, N, IC, LOC, sh, sw, ph, pw, GN, GM);

					//ksIms_42R(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, N, IC, OC, sh, sw, ph, pw, GN, GM);
					//ksIms_24R(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, N, IC, OC, sh, sw, ph, pw, GN, GM);
					//ksIms_22R(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, N, IC, OC, sh, sw, ph, pw, GN, GM);
					//ksIms_21R(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, N, IC, OC, sh, sw, ph, pw, GN, GM);
					//ksIms_12R(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, N, IC, OC, sh, sw, ph, pw, GN, GM);
					//ksIms_11R(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, N, IC, OC, sh, sw, ph, pw, GN, GM);
				}
			}

			//KernelSplit Ims2R==================================================
			{
				//Ims2_init(N, IH, IW, FH, FW, OC, IC);
				
				//__ks_remodev2(NULL, dW, FH, FW, dCW, CFH, CFW, OC, IC, sh, sw);

				//V2 uernel
				{
					//ksV2_Ims2_u88R(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, N);
					//ksV2_Ims2_u88R_CFW_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, LOG2(OC), ph, pw, GN, N);
					//ksV2_Ims2_u88R_W5_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, d_deltaX, IH_slice, IW_slice, IC, LOG2(OC), ph, pw, GN, N);
				}

				//V2 kenrel
				{
					//ksV2_Ims2_88R_CFW_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, LOG2(OC), ph, pw, GN, N);
					//ksV2_Ims2_88R_W5_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, d_deltaX, IH_slice, IW_slice, IC, LOG2(OC), ph, pw, GN, N);

					//ksV2_Ims2_88R(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, N);
					//ksV2_Ims2_84R(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, N);
					//ksV2_Ims2_48R(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, N);
					//ksV2_Ims2_44R(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, N);
				}
				
				//uernel
				{
					//ksIms2_u88R8_oc_CFW2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, LOG2(OC), ph, pw, GN, GM);
					//ksIms2_u88R4_oc_CFW2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, LOG2(OC), ph, pw, GN, GM);
					//ksIms2_u88R_oc_CFW2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, LOG2(OC), ph, pw, GN, GM);
				}

				//cfw oc 2pow
				{
					//ksIms2_88R8_oc_CFW2pow_tex(NULL, LB, 0, 0, texDy, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, LOG2(OC), ph, pw, GN, GM);
					//ksIms2_88R4_oc_CFW2pow_tex(NULL, LB, 0, 0, texDy, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, LOG2(OC), ph, pw, GN, GM);

					//ksIms2_88R8_oc_CFW2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, LOG2(OC), ph, pw, GN, GM);
					//ksIms2_88R4_oc_CFW2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, LOG2(OC), ph, pw, GN, GM);
					//ksIms2_88R_oc_CFW2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, LOG2(OC), ph, pw, GN, GM);
				}

				//oc 2pow
				{
					//ksIms2_88R8_oc2pow_tex(NULL, LB, 0, 0, texDy, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, LOG2(OC), ph, pw, GN, GM);
					//ksIms2_88R4_oc2pow_tex(NULL, LB, 0, 0, texDy, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, LOG2(OC), ph, pw, GN, GM);
					//ksIms2_88R_oc2pow_tex(NULL, LB, 0, 0, texDy, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, LOG2(OC), ph, pw, GN, GM);

					//ksIms2_88R8_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, LOG2(OC), ph, pw, GN, GM);
					//ksIms2_88R4_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, LOG2(OC), ph, pw, GN, GM);
					//ksIms2_88R_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, LOG2(OC), ph, pw, GN, GM);
				}
				
				//CW 2pow 
				{
					//ksIms2_88R8_CW2pow_tex(NULL, LB, 0, 0, texDy, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, GM);
					//ksIms2_88R4_CW2pow_tex(NULL, LB, 0, 0, texDy, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, GM);

					//ksIms2_88R8_CW2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, GM);
					//ksIms2_88R4_CW2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, GM);
					//ksIms2_88R_CW2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, GM);
				}

				//common 8*8
				{
					//ksIms2_88R8_tex(NULL, LB, 0, 0, texDy, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, GM);
					//ksIms2_88R4_tex(NULL, LB, 0, 0, texDy, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, GM);
					//ksIms2_88R_tex(NULL, LB, 0, 0, texDy, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, GM);

					//ksIms2_88R8(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, GM);
					//ksIms2_88R4(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, GM);
					//ksIms2_88R(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, GM);
				}

				//Kernel A
				{
					//ksIms2A_88R(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, N, IC, OC, ph, pw, GN, GM);
					//ksIms2A_88R_CW2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, N, IC, OC, ph, pw, GN, GM);
					//ksIms2A_88R_oc_CFW2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, N, IC, LOG2(OC), ph, pw, GN, GM);
				}
				
				//others
				{
					//ksIms2_84R_tex(NULL, LB, 0, 0, texDy, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, GM);
					//ksIms2_84R_oc2pow_tex(NULL, LB, 0, 0, texDy, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, LOC, ph, pw, GN, GM);

					//ksIms2_48R_tex(NULL, LB, 0, 0, texDy, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, GM);
					//ksIms2_48R_oc2pow_tex(NULL, LB, 0, 0, texDy, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, LOC, ph, pw, GN, GM);

					//ksIms2_44R_tex(NULL, LB, 0, 0, texDy, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, GM);
					//ksIms2_44R_oc2pow_tex(NULL, LB, 0, 0, texDy, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, LOC, ph, pw, GN, GM);

					//ksIms2_84R(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, GM);
					//ksIms2_84R_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, LOC, ph, pw, GN, GM);
					//ksIms2_48R(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, GM);
					//ksIms2_48R_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, LOC, ph, pw, GN, GM);
					//ksIms2_44R(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, GM);
					//ksIms2_44R_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, LOC, ph, pw, GN, GM);
					//ksIms2_42R(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, GM);
					//ksIms2_24R(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, GM);
				}
				
			}

			//Kernel Split=======================================================
			{{
				//KS_init(N, IH, IW, FH, FW, OC, IC, sh, sw);
				//__ks_remodev2(NULL, dW, FH, FW, dCW, CFH, CFW, OC, IC, sh, sw);

				//ks88R_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH, IW, N, IC, LOG2(OC), sh, sw, ph, pw, GN, GM);
				//ks84R_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH, IW, N, IC, LOG2(OC), sh, sw, ph, pw, GN, GM);
				//ks48R_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH, IW, N, IC, LOG2(OC), sh, sw, ph, pw, GN, GM);
				//ks44R_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH, IW, N, IC, LOG2(OC), sh, sw, ph, pw, GN, GM);

				//ks88R(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, GN, GM);
				//ks84R(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, GN, GM);
				//ks48R(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, GN, GM);
				//ks44R(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, GN, GM);
				//ks42R(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, GN, GM);
				//ks24R(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, GN, GM);
				//ks22R(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, GN, GM);
				//ks21R(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, GN, GM);
				//ks12R(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, GN, GM);
				//ks11R(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, GN, GM);
			}}
		}
		
		//Cross Add==============================================================
		{{	
			//__kernel_remode(NULL, dW, dCW, FH, FW, OC, IC);
			//cudaMemsetAsync(d_deltaX, 0, sizeof(float)*sizeX, NULL);

			//vop_kernel1(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, OC, sh, sw, ph, pw, OC, (N*OH*OW));
			//vop_kernel2(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, OC, sh, sw, ph, pw, OC, (N*OH*OW));
			//vop_kernel3(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, OC, sh, sw, ph, pw, OC, (N*OH*OW));
			//vop_kernel4(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, OC, sh, sw, ph, pw, OC, (N*OH*OW));
			//vop_kernel5(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, OC, sh, sw, ph, pw, OC, (N*OH*OW));
			//vop_kernel6(NULL, LB, 0, 0, d_deltaY, OH, OW, dCW, FH, FW, d_deltaX, IH, IW, IC, OC, sh, sw, ph, pw, OC, (N*OH*OW));
			//vop_kernel7(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, 3, 3, d_deltaX, IH, IW, 16, OC, sh, sw, ph, pw, OC, (N*OH*OW));
			//vop_kernel8(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, OC, sh, sw, ph, pw, OC, (N*OH*OW));
			//vop_kernel9(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, OC, sh, sw, ph, pw, OC, (N*OH*OW));
			//vop_kernel10(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, OC, sh, sw, ph, pw, OC, (N*OH*OW));
			//vop_kernel11(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, OC, sh, sw, ph, pw, OC, (N*OH*OW));

			//int index = 0; __dconv3D_deltaX_CrossAdd(streams, index, 8, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw);
			//crossAdd_k16_2(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, OC, sh, sw, ph, pw, OC, (N*OH*OW));
			//crossAdd_k82(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, OC, sh, sw, ph, pw, OC, (N*OH*OW));
			//crossAdd_k42(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, OC, sh, sw, ph, pw, OC, (N*OH*OW));
			//crossAdd_k22(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, OC, sh, sw, ph, pw, OC, (N*OH*OW));
			//crossAdd_k11(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, OC, sh, sw, ph, pw, OC, (N*OH*OW));
		}}

		//Winograd2D=============================================================
		{
			//int GN = IC;
			//int GM = N * (IH >> 1) * (IW >> 1);

			//__deconv3D_dX_winograd2D_f22x33_kernel_remode_v2(stream, dW, dG, OC, IC);
			//__deconv3D_input_pad_remode(NULL, d_deltaY, OH, OW, dD, DH, DW, N, OC, ph, pw);
			//deconv3d_dX_Winograd2d_f22x33_k32x32R(NULL, 0, 0, dD, DH, DW, dG, d_deltaX, IH, IW, N, OC, IC, GN, GM);
		}

		//Img2col Winograd=======================================================
		{
			//------[FH = FW = 2]------------------------------------------------
			{
				//winograd_f3x2_k64x192C_tex(NULL, 0, texDy, OH, OW, dW, 2, d_deltaX, IH, IW, IC, OC, ph, pw, IW);
				//winograd_f3x2_k64x192x6C_tex(NULL, 0, texDy, OH, OW, dW, 2, d_deltaX, IH, IW, IC, OC, ph, pw, IW);

				//FW % 2 == 0
				//winograd_SFW_f3x2_k64x192C_tex(NULL, 0, texDy, OH, OW, dW, FH, 4, d_deltaX, IH, IW, IC, OC, ph, pw, IW);

				//==============================================================================
				//winograd_f7x2_k64x224_tex(NULL, texDy, OH, OW, dW, 2, d_deltaX, IH, IW, N, IC, OC, ph, pw);
				//winograd_f7x2_k64x224_p0_tex(NULL, texDy, OH, OW, dW, 2, d_deltaX, IH, IW, N, IC, OC, ph, pw);

				/*int GNr, GMr, index = 0;
				deconv3D_dX_winograd_s8_W2_64x32R_tex<2>(streams, index, length,
					d_deltaY, texDy, OH, OW, dW, d_deltaX, IH, IW, N, IC, OC, ph, pw,
					GN, GM, GMr, GNr);*/
			}

			//------[FH = FW = 3]------------------------------------------------
			{
				//winograd_f2x3_k64x128C_tex(NULL, 0, texDy, OH, OW, dW, 3, d_deltaX, IH, IW, IC, OC, ph, pw, IW);
				//winograd_f2x3_k64x128x4C_tex(NULL, 0, texDy, OH, OW, dW, 3, d_deltaX, IH, IW, IC, OC, ph, pw, IW);

				//winograd_f6x3_k64x192_tex(NULL, texDy, OH, OW, dW, 3, d_deltaX, IH, IW, N, IC, OC, ph, pw);
				//winograd_f6x3_k64x192_p1_tex(NULL, texDy, OH, OW, dW, 3, d_deltaX, IH, IW, N, IC, OC, ph, pw);
				//winograd_f6x3_k64x192_p1_ICT_tex(NULL, texDy, OH, OW, dW, 3, d_deltaX, IH, IW, N, OC, 128, ph, pw);

				/*int GNr, GMr, index = 0;
				deconv3D_dX_winograd_s8_W3_64x32R_tex<3>(streams, index, length,
					d_deltaY, texDy, OH, OW, dW, d_deltaX, IH, IW, N, IC, OC, ph, pw,
					GN, GM, GMr, GNr);*/

				//FW % 3 == 0
				//winograd_SFW_f2x3_k64x128C_tex(NULL, 0, texDy, OH, OW, dW, FH, 6, d_deltaX, IH, IW, IC, OC, ph, pw, IW);
				//winograd_SFW_f2x3_k64x128x4C_tex(NULL, 0, texDy, OH, OW, dW, FH, 6, d_deltaX, IH, IW, IC, OC, ph, pw, IW);
			}

			//------[FH = FW = 4]------------------------------------------------
			{
				//winograd_f5x4_k64x160_tex(NULL, texDy, OH, OW, dW, 4, d_deltaX, IH, IW, N, IC, OC, ph, pw);
				//winograd_f5x4_k64x160_p1_tex(NULL, texDy, OH, OW, dW, 4, d_deltaX, IH, IW, N, IC, OC, ph, pw);

				//==================================================
				/*int GNr, GMr, index = 0;
				deconv3D_dX_winograd_s8_W4_64x32R_tex<4>(streams, index, length,
					d_deltaY, texDy, OH, OW, dW, d_deltaX, IH, IW, N, IC, OC, ph, pw,
					GN, GM, GMr, GNr);*/

				//======[FW % 4 == 0]===============================
				//winograd_SFW_f5x4_k64x160C_tex(NULL, 0, texDy, OH, OW, dW, FH, 8, d_deltaX, IH, IW, N, IC, OC, ph, pw, IW);
			}

			//------[FH = FW = 5]------------------------------------------------
			{
				//winograd_f4x5_k64x128_tex(NULL, texDy, OH, OW, dW, 5, d_deltaX, IH, IW, N, IC, OC, ph, pw);
				//winograd_f4x5_k64x128_p2_tex(NULL, texDy, OH, OW, dW, 5, d_deltaX, IH, IW, N, IC, OC, ph, pw);
				//winograd_f4x5_k64x128C_p2_tex(NULL, 0, texDy, OH, OW, dW, 5, d_deltaX, IH, IW, N, IC, OC, ph, pw, IW);
				//winograd_f4x5_ruse_k64x128_p2_tex(NULL, texDy, OH, OW, dW, 5, d_deltaX, IH, IW, N, IC, OC, ph, pw);

				/*int GNr, GMr, index = 0;
				deconv3D_dX_winograd_s8_W5_64x32R_tex<5>(streams, index, length,
					d_deltaY, texDy, OH, OW, dW, d_deltaX, IH, IW, N, IC, OC, ph, pw,
					GN, GM, GMr, GNr);*/
			}

			//------[FH = FW = 6]------------------------------------------------
			{
				//winograd_f3x6_k64x96_tex(NULL, texDy, OH, OW, dW, 6, d_deltaX, IH, IW, N, IC, OC, ph, pw);
				//winograd_f3x6_k64x96_ICT_tex(NULL, texDy, OH, OW, dW, 6, d_deltaX, IH, IW, N, 256, OC, ph, pw);
				//winograd_f3x6_ruse_k64x96_p2_tex(NULL, texDy, OH, OW, dW, 6, d_deltaX, IH, IW, N, IC, OC, ph, pw);

				//winograd_f3x6_k64x96C_tex(NULL, 0, texDy, OH, OW, dW, 6, d_deltaX, IH, IW, N, IC, OC, ph, pw, IW);
				//winograd_f3x6_k64x96C_CT_tex(NULL, 0, texDy, OH, OW, dW, 6, d_deltaX, IH, IW, N, 64, 64, ph, pw, IW);

				/*int GNr, GMr, index = 0;
				deconv3D_dX_winograd_s8_W6_64x32R_tex<6>(streams, index, length,
					d_deltaY, texDy, OH, OW, dW, d_deltaX, IH, IW, N, IC, OC, ph, pw,
					GN, GM, GMr, GNr);*/
			}

			//------[FH = FW = 7]------------------------------------------------
			{
				//===============================================================
				//winograd_f2x7_k64x64_tex(NULL, texDy, OH, OW, dW, 7, d_deltaX, IH, IW, N, IC, OC, ph, pw);

				//winograd_f2x7_k64x64C_tex(NULL, 0, texDy, OH, OW, dW, 7, d_deltaX, IH, IW, N, IC, OC, ph, pw, IW);
				//winograd_f2x7_k64x64C_CT_tex(NULL, 0, texDy, OH, OW, dW, 7, d_deltaX, IH, IW, N, 256, 256, ph, pw, IW);

				//winograd_f2x7_ruse_k64x64_p3_tex(NULL, texDy, OH, OW, dW, 7, d_deltaX, IH, IW, N, IC, OC, ph, pw);
				//winograd_f2x7_ruse_k64x64_p3_CT_tex(NULL, texDy, OH, OW, dW, 7, d_deltaX, IH, IW, N, 512, 512, ph, pw);

				//winograd_f2x7_ruse_k64x64C_p3_tex(NULL, 0, texDy, OH, OW, dW, 7, d_deltaX, IH, IW, N, IC, OC, ph, pw, IW);
				//winograd_f2x7_ruse_k64x64C_p3_CT_tex(NULL, 0, texDy, OH, OW, dW, 7, d_deltaX, IH, IW, N, 128, 128, ph, pw, IW);

				/*int GNr, GMr, index = 0;
				deconv3D_dX_winograd_s8_W7_64x32R_tex<7>(streams, index, length,
					d_deltaY, texDy, OH, OW, dW, d_deltaX, IH, IW, N, IC, OC, ph, pw,
					GN, GM, GMr, GNr);*/

				//================================================================
				//winograd_f10x7_k32x320_p3_tex(NULL, texDy, OH, OW, dW, 7, d_deltaX, IH, IW, IC, OC, ph, pw);

				/*int GNr, GMr, index = 0;
				deconv3D_dX_Winograd_s16_W7_32x32_p3_tex<7>(streams, index, length,
					d_deltaY, texDy, OH, OW, dW, d_deltaX, IH, IW, N, IC, OC, ph, pw,
					GN, GM, GMr, GNr);*/

				//==================================================
				//winograd_f10x7_k64x320_p3_tex(NULL, texDy, OH, OW, dW, 7, d_deltaX, IH, IW, N, IC, OC, ph, pw);
				//winograd_f10x7_k64x320_p3_OCT_tex(NULL, texDy, OH, OW, dW, 7, d_deltaX, IH, IW, N, IC, 64, ph, pw);

				/*int GNr, GMr, index = 0;
				deconv3D_dX_Winograd_s16_W7_64x32_p3_tex<7>(streams, index, length,
					d_deltaY, texDy, OH, OW, dW, d_deltaX, IH, IW, N, IC, OC, ph, pw,
					GN, GM, GMr, GNr);*/
			}

			//------[FH = FW = 8]------------------------------------------------
			{
				//==================================================
				//winograd_f9x8_k32x288_p4_tex(NULL, texDy, OH, OW, dW, 8, d_deltaX, IH, IW, N, IC, OC, ph, pw);
				//winograd_f9x8_k32x288_p4_ICT_tex(NULL, texDy, OH, OW, dW, 8, d_deltaX, IH, IW, N, 256, OC, ph, pw);
				//winograd_f9x8_k32x288_p4_CT_tex(NULL, texDy, OH, OW, dW, 8, d_deltaX, IH, IW, N, 128, 128, ph, pw);
				//winograd_f9x8_ruse_k32x288_p4_tex(NULL, texDy, OH, OW, dW, 8, d_deltaX, IH, IW, N, IC, OC, ph, pw);

				/*int GNr, GMr, index = 0;
				deconv3D_dX_Winograd_s16_W8_32x32_p3_tex<8>(streams, index, length,
					d_deltaY, texDy, OH, OW, dW, d_deltaX, IH, IW, N, IC, OC, ph, pw,
					GN, GM, GMr, GNr);*/

				//==================================================
				//winograd_f9x8_k64x288_p4_tex(NULL, texDy, OH, OW, dW, 8, d_deltaX, IH, IW, N, IC, OC, ph, pw);
				//winograd_f9x8_k64x288_p4_OCT_tex(NULL, texDy, OH, OW, dW, 8, d_deltaX, IH, IW, N, IC, 256, ph, pw);

				int GNr, GMr, index = 0;
				deconv3D_dX_Winograd_s16_W8_64x32_p3_tex<8>(streams, index, length,
					d_deltaY, texDy, OH, OW, dW, d_deltaX, IH, IW, N, IC, OC, ph, pw,
					GN, GM, GMr, GNr);
			}

			//------[FH = FW = 9]------------------------------------------------
			{
				//==================================================
				//winograd_f8x9_k32x256_p4_tex(NULL, texDy, OH, OW, dW, 9, d_deltaX, IH, IW, N, IC, OC, ph, pw);
				//winograd_f8x9_k32x256_p4_ICT_tex(NULL, texDy, OH, OW, dW, 9, d_deltaX, IH, IW, N, 256, OC, ph, pw);
				//winograd_f8x9_ruse_k32x256_p4_tex(NULL, texDy, OH, OW, dW, 9, d_deltaX, IH, IW, N, IC, OC, ph, pw);

				/*int GNr, GMr, index = 0;
				deconv3D_dX_Winograd_s16_W9_32x32_p4_tex<9>(streams, index, length,
					d_deltaY, texDy, OH, OW, dW, d_deltaX, IH, IW, N, IC, OC, ph, pw,
					GN, GM, GMr, GNr);*/

				//==================================================
				//winograd_f8x9_k64x256_p4_tex(NULL, texDy, OH, OW, dW, 9, d_deltaX, IH, IW, N, IC, OC, ph, pw);
				//winograd_f8x9_k64x256_p4_OCT_tex(NULL, texDy, OH, OW, dW, 9, d_deltaX, IH, IW, N, IC, 64, ph, pw);

				/*int GNr, GMr, index = 0;
				deconv3D_dX_Winograd_s16_W9_64x32_p4_tex<9>(streams, index, length,
					d_deltaY, texDy, OH, OW, dW, d_deltaX, IH, IW, N, IC, OC, ph, pw,
					GN, GM, GMr, GNr);*/
			}

			//__winograd_f2x3_k48_tex<LB>(NULL, 0, 0, texDy, OH, OW, dW, d_deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
		}

		//ZeroPadding s1=========================================================
		{{
			//uernel V2
			{
				//uV2_88s1(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, OC, ph, pw, GN, N);
				//uV2_88s1_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, LOG2(OC), ph, pw, GN, N);

				//uV2_88s1W3P1(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, d_deltaX, IH, IW, IC, OC, GN, N);
				//uV2_88s1W3P1_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, d_deltaX, IH, IW, IC, LOG2(OC), GN, N);

				//uV2_88s1W5P2(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, d_deltaX, IH, IW, IC, OC, GN, N);
				//uV2_88s1W5P2_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, d_deltaX, IH, IW, IC, LOG2(OC), GN, N);
			}

			//kernel V2
			{
				//kV2_88s1W5P2(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, d_deltaX, IH, IW, IC, OC, GN, N);
				//kV2_88s1W5P2_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, d_deltaX, IH, IW, IC, LOG2(OC), GN, N);
				
				//kV2_88s1W3P1(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, d_deltaX, IH, IW, IC, OC, GN, N);
				//kV2_88s1W3P1_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, d_deltaX, IH, IW, IC, LOG2(OC), GN, N);

				//kV2_88s1_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, LOG2(OC), ph, pw, GN, N);
				//kV2_84s1_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, LOG2(OC), ph, pw, GN, N);
				//kV2_48s1_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, LOG2(OC), ph, pw, GN, N);
				//kV2_44s1_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, LOG2(OC), ph, pw, GN, N);

				//kV2_88s1(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, OC, ph, pw, GN, N);
				//kV2_84s1(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, OC, ph, pw, GN, N);
				//kV2_48s1(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, OC, ph, pw, GN, N);
				//kV2_44s1(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, OC, ph, pw, GN, N);
			}
			
			//kernel A
			{
				//k88As1W3(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, d_deltaX, IH, IW, N, IC, OC, ph, pw, GN, GM);
				//k88As1W3_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, d_deltaX, IH, IW, N, IC, LOG2(OC), ph, pw, GN, GM);
				//k88As1W3_oc2pow_tex(NULL, LB, 0, 0, texDy, OH, OW, dW, d_deltaX, IH, IW, N, IC, LOG2(OC), ph, pw, GN, GM);
				
				//k88As1W5(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, d_deltaX, IH, IW, N, IC, OC, ph, pw, GN, GM);
				//k88As1W5_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, d_deltaX, IH, IW, N, IC, LOG2(OC), ph, pw, GN, GM);
				//k88As1W5_oc2pow_tex(NULL, LB, 0, 0, texDy, OH, OW, dW, d_deltaX, IH, IW, N, IC, LOG2(OC), ph, pw, GN, GM);

				//k88As1_tex(NULL, LB, 0, 0, texDy, OH, OW, dW, FH, FW, d_deltaX, IH, IW, N, IC, OC, ph, pw, GN, GM);
				//k88As1_oc2pow_tex(NULL, LB, 0, 0, texDy, OH, OW, dW, FH, FW, d_deltaX, IH, IW, N, IC, LOG2(OC), ph, pw, GN, GM);
				
				//k88As1(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, N, IC, OC, ph, pw, GN, GM);
				//k88As1_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, N, IC, LOG2(OC), ph, pw, GN, GM);
			}

			//uernel C
			{
				//u88s1C(NULL, LB, 0, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
				//u88s1x2C(NULL, LB, 0, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, OC, ph, pw, GN, GM);

				//u88s1x2C_oc2pow(NULL, LB, 0, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, LOG2(OC), ph, pw, GN, GM);

				//u84s1C_pure(NULL, LB, 0, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
				//u48s1C_pure(NULL, LB, 0, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
				//u44s1C_pure(NULL, LB, 0, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
			}

			//uernel ruse
			{
				//u88s1x4_ruse(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, 3, 3, d_deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
				//u88s1x4_ruse(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, 9, 9, d_deltaX, IH, IW, IC, OC, ph, pw, GN, GM);

				//u88s1W3x4_ruse(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, d_deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
			}

			//uernel
			{
				//u88s1(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
				//u88As1(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, N, IC, OC, ph, pw, GN, GM);
				//u88As1_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, N, IC, LOG2(OC), ph, pw, GN, GM);

				//u88As1W3(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, d_deltaX, IH, IW, N, IC, OC, ph, pw, GN, GM);
				//u88As1W3_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, d_deltaX, IH, IW, N, IC, LOG2(OC), ph, pw, GN, GM);
				//u88s1W3x4_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, d_deltaX, IH, IW, IC, LOG2(OC), ph, pw, GN, GM);

				//u88As1W5(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, d_deltaX, IH, IW, N, IC, OC, ph, pw, GN, GM);
				//u88As1W5_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, d_deltaX, IH, IW, N, IC, LOG2(OC), ph, pw, GN, GM);
				//u88s1W5x4_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, d_deltaX, IH, IW, IC, LOG2(OC), ph, pw, GN, GM);

				//u84s1_pure(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
				//u48s1_pure(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
				//u44s1_pure(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
			}

			//kernel
			{
				//f88s1(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, 3, 3, d_deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
				//f88s1(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, 5, 5, d_deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
				//f88s1x4(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, 3, 3, d_deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
				//f88s1x4(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, 5, 5, d_deltaX, IH, IW, IC, OC, ph, pw, GN, GM);

				//f88s1x4_tex(NULL, LB, 0, 0, texDy, OH, OW, dW, 3, 3, d_deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
				//f88s1x4_tex(NULL, LB, 0, 0, texDy, OH, OW, dW, 5, 5, d_deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
				//f88s1_tex(NULL, LB, 0, 0, texDy, OH, OW, dW, 3, 3, d_deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
				//f88s1_tex(NULL, LB, 0, 0, texDy, OH, OW, dW, 5, 5, d_deltaX, IH, IW, IC, OC, ph, pw, GN, GM);

				//==================================================================
				//k88s1W5x4_oc2pow_tex(NULL, LB, 0, 0, texDy, OH, OW, dW, d_deltaX, IH, IW, IC, LOG2(OC), ph, pw, GN, GM);
				//k88s1W5_oc2pow_tex(NULL, LB, 0, 0, texDy, OH, OW, dW, d_deltaX, IH, IW, IC, LOG2(OC), ph, pw, GN, GM);
				
				//k88s1W5x4_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, d_deltaX, IH, IW, IC, LOG2(OC), ph, pw, GN, GM);
				//k88s1W5_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, d_deltaX, IH, IW, IC, LOG2(OC), ph, pw, GN, GM);
	
				//==================================================================
				//k88s1W3x4_oc2pow_tex(NULL, LB, 0, 0, texDy, OH, OW, dW, d_deltaX, IH, IW, IC, LOG2(OC), ph, pw, GN, GM);
				//k88s1W3_oc2pow_tex(NULL, LB, 0, 0, texDy, OH, OW, dW, d_deltaX, IH, IW, IC, LOG2(OC), ph, pw, GN, GM);
				
				//k88s1W3x4_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, d_deltaX, IH, IW, IC, LOG2(OC), ph, pw, GN, GM);
				//k88s1W3_oc2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, d_deltaX, IH, IW, IC, LOG2(OC), ph, pw, GN, GM);
			
				//==================================================================
				//k88s1_W2pow_tex(NULL, LB, 0, 0, texDy, OH, OW, dW, LOG2(FH), LOG2(FW), d_deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
				//k88s1_tex(NULL, LB, 0, 0, texDy, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, OC, ph, pw, GN, GM);

				//k88s1_W2pow(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, log2(FH), log2(FW), d_deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
				//k88s1(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, OC, ph, pw, GN, GM);

				//==================================================================
				//k44s1(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
				//k22s1(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
				//k21s1(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
				//k12s1(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
				//k11s1(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
			}

			//pure kernel
			{
				//k48s1_pure_tex(NULL, LB, 0, 0, texDy, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
				//k44s1_pure_tex(NULL, LB, 0, 0, texDy, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
				
				//k84s1_pure(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
				//k48s1_pure(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
				//k44s1_pure(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
				//k82s1_pure(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
				//k28s1_pure(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
				//k42s1_pure(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
				//k24s1_pure(NULL, LB, 0, 0, d_deltaY, OH, OW, dW, FH, FW, d_deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
			}
		}}

		//Kernel W1==============================================================
		{{
			//k88W1_LB4(NULL, 0, 0, d_deltaY, dW, d_deltaX, IC, OC, GN, GM);
			//k88W1_LB3(NULL, 0, 0, d_deltaY, dW, d_deltaX, IC, OC, GN, GM);
			//k84W1(NULL, LB, 0, 0, d_deltaY, dW, d_deltaX, IC, OC, GN, GM);
			//k48W1(NULL, LB, 0, 0, d_deltaY, dW, d_deltaX, IC, OC, GN, GM);
			//k44W1(NULL, LB, 0, 0, d_deltaY, dW, d_deltaX, IC, OC, GN, GM);
			//k82W1(NULL, LB, 0, 0, d_deltaY, dW, d_deltaX, IC, OC, GN, GM);
			//k28W1(NULL, LB, 0, 0, d_deltaY, dW, d_deltaX, IC, OC, GN, GM);
			//k42W1(NULL, LB, 0, 0, d_deltaY, dW, d_deltaX, IC, OC, GN, GM);
			//k24W1(NULL, LB, 0, 0, d_deltaY, dW, d_deltaX, IC, OC, GN, GM);
			//k22W1(NULL, LB, 0, 0, d_deltaY, dW, d_deltaX, IC, OC, GN, GM);
			//k21W1(NULL, LB, 0, 0, d_deltaY, dW, d_deltaX, IC, OC, GN, GM);
			//k12W1(NULL, LB, 0, 0, d_deltaY, dW, d_deltaX, IC, OC, GN, GM);
			//k11W1(NULL, LB, 0, 0, d_deltaY, dW, d_deltaX, IC, OC, GN, GM);
			//int index = 0; __dconv3D_deltaX_W1(streams, index, 8, d_deltaY, dW, d_deltaX, IH, IW, IC, OC);
		}}
		//-------------------------------------
		error = cudaDeviceSynchronize(); printError(error);
	}

	error = cudaDestroyTextureObject(texDy); printError(error);
	error = cudaGetLastError(); printError(error);
	error = cudaDeviceSynchronize(); printError(error);
	error = cudaGetLastError(); printError(error);
	clock_t end = clock();

	int div = end - start;
	float time = 1.0f*div / nIter;

	float size = 1.0f * GN / 1024 * GM / 1024 * GK / 1024;
	float total = 2 * 1024 * size * 1e-9f * 1024 * 1024;
	float performance = total / (time*1e-3f);
	cout << "Size = " << size << ", Time = " << time << " msec, Performace = " << performance << " GFlop/s";

	long long Rsize = sizeW + sizeWks;
	//long long Rsize = 2.78 * sizeW ;
	//long long Rsize = 2 * sizeX;

	float Rspeed = (1.0f *(Rsize) / (1 << 28)) / (time*1e-3);
	cout << endl << "Rsize = " << 1.0f * Rsize / 1024 / 1024 << endl;
	cout << endl << "Rspeed = " << Rspeed << " GB/s" << endl;

	error = cudaFree(d_deltaX); printError(error);
	error = cudaFree(d_deltaY); printError(error);

	delete deltaY;
	//delete W;
}


int maxInt(int a, int b)
{
	return b & ((a - b) >> 31) | a & (~(a - b) >> 31);
}

void test()
{
	goto stride1;
	//=======[Down Sampling Area]==========================================
	{
		//int FH = 3, FW = 3, ph = 1, pw = 1, sh = 2, sw = 2;
		//int IH = 64, IW = 64, OH = 32, OW = 32, N = 128, IC = 128, OC = 128;
		//int IH = 32, IW = 32, OH = 16, OW = 16, N = 128, IC = 256, OC = 256;
		//int IH = 28, IW = 28, OH = 14, OW = 14, N = 128, IC = 256, OC = 256;
		//int IH = 24, IW = 24, OH = 12, OW = 12, N = 128, IC = 256, OC = 256;
		//int IH = 16, IW = 16, OH = 8, OW = 8, N = 128, IC = 512, OC = 512;
		//int IH = 16, IW = 16, OH = 8, OW = 8, N = 512, IC = 128, OC = 256;
		//int IH = 12, IW = 12, OH = 6, OW = 6, N = 512, IC = 256, OC = 256;
		//int IH = 8, IW = 8, OH = 4, OW = 4, N = 512, IC = 512, OC = 512;
		//int IH = 4, IW = 4, OH = 2, OW = 2, N = 512, IC = 512, OC = 1024;

		//int FH = 5, FW = 5, ph = 2, pw = 2, sh = 2, sw = 2;
		//int IH = 64, IW = 64, OH = 32, OW = 32, N = 128, IC = 128, OC = 64;
		//int IH = 32, IW = 32, OH = 16, OW = 16, N = 128, IC = 128, OC = 128;
		//int IH = 28, IW = 28, OH = 14, OW = 14, N = 128, IC = 128, OC = 128;
		//int IH = 24, IW = 24, OH = 12, OW = 12, N = 128, IC = 128, OC = 256;
		//int IH = 16, IW = 16, OH = 8, OW = 8, N = 128, IC = 256, OC = 256;
		//int IH = 12, IW = 12, OH = 6, OW = 6, N = 128, IC = 256, OC = 256;
		//int IH = 8, IW = 8, OH = 4, OW = 4, N = 128, IC = 512, OC = 512;
		//int IH = 4, IW = 4, OH = 2, OW = 2, N = 512, IC = 512, OC = 512;

		//testCorrect<3>(IH, IW, OH, OW, FH, FW, N/4, IC/2, OC/4, sh, sw, ph, pw);//3*3*4 = 9*4=36
		//testCorrect<4>(IH/2, IW/2, OH/2, OW/2, FH, FW, N/2, IC, OC/4, sh, sw, ph, pw);//3*3*4 = 9*4=36

		//testCorrect<4>(IH, IW, OH, OW, FH, FW, N/4, IC/4, OC/4, sh, sw, ph, pw);//3*3*4 = 9*4=36
		//testCorrect<4>(IH/2, IW/2, OH/2, OW/2, FH, FW, N/4, IC, OC/4, sh, sw, ph, pw);//3*3*4 = 9*4=36
		//testSpeed<4>(1000, IH, IW, OH, OW, FH, FW, N, IC, OC, sh, sw, ph, pw);//3*3*2 = 9*2
	}

stride1: 
	//======[Area for kernel S1]======================================
	{
		//int FH = 2, FW = 2, ph = 1, pw = 1, sh = 1, sw = 1;
		//int FH = 3, FW = 3, ph = 1, pw = 1, sh = 1, sw = 1;
		//int FH = 4, FW = 4, ph = 2, pw = 2, sh = 1, sw = 1;
		//int FH = 5, FW = 5, ph = 2, pw = 2, sh = 1, sw = 1;
		//int FH = 6, FW = 6, ph = 3, pw = 3, sh = 1, sw = 1;
		//int FH = 7, FW = 7, ph = 3, pw = 3, sh = 1, sw = 1;
		int FH = 8, FW = 8, ph = 4, pw = 4, sh = 1, sw = 1;
		//int FH = 9, FW = 9, ph = 4, pw = 4, sh = 1, sw = 1;

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

		//======[4 * 4]=======================================================
		//int IH = 80, IW = 80, OH = 80, OW = 80, N = 128, IC = 64, OC = 64;
		//int IH = 40, IW = 40, OH = 40, OW = 40, N = 128, IC = 128, OC = 128;
		//int IH = 20, IW = 20, OH = 20, OW = 20, N = 128, IC = 256, OC = 256;
		//int IH = 10, IW = 10, OH = 10, OW = 10, N = 128, IC = 512, OC = 512;

		//int IH = 128, IW = 128, OH = 128, OW = 128, N = 32, IC = 64, OC = 64;
		//int IH = 64, IW = 64, OH = 64, OW = 64, N = 128, IC = 64, OC = 64;
		//int IH = 32, IW = 32, OH = 32, OW = 32, N = 128, IC = 128, OC = 128;
		//int IH = 16, IW = 16, OH = 16, OW = 16, N = 128, IC = 256, OC = 256;
		//int IH = 8, IW = 8, OH = 8, OW = 8, N = 128, IC = 512, OC = 512;
		//int IH = 24, IW = 24, OH = 24, OW = 24, N = 128, IC = 256, OC = 256;

		//======[5 * 5]=======================================================
		//int IH = 128, IW = 128, OH = 128, OW = 128, N = 32, IC = 64, OC = 64;
		//int IH = 64, IW = 64, OH = 64, OW = 64, N = 64, IC = 128, OC = 128;
		//int IH = 32, IW = 32, OH = 32, OW = 32, N = 128, IC = 128, OC = 128;
		//int IH = 16, IW = 16, OH = 16, OW = 16, N = 64, IC = 512, OC = 512;

		//======[6 * 6]=======================================================
		//int IH = 96, IW = 96, OH = 96, OW = 96, N = 64, IC = 64, OC = 64;
		//int IH = 48, IW = 48, OH = 48, OW = 48, N = 128, IC = 64, OC = 64;
		//int IH = 24, IW = 24, OH = 24, OW = 24, N = 128, IC = 128, OC = 128;
		//int IH = 12, IW = 12, OH = 12, OW = 12, N = 64, IC = 256, OC = 256;

		//int IH = 128, IW = 128, OH = 128, OW = 128, N = 32, IC = 64, OC = 64;
		//int IH = 32, IW = 32, OH = 32, OW = 32, N = 128, IC = 128, OC = 128;
		//int IH = 16, IW = 16, OH = 16, OW = 16, N = 128, IC = 256, OC = 256;
		//int IH = 8, IW = 8, OH = 8, OW = 8, N = 128, IC = 512, OC = 512;

		//======[7 * 7]=======================================================
		//int IH = 128, IW = 128, OH = 128, OW = 128, N = 32, IC = 64, OC = 64;
		//int IH = 64, IW = 64, OH = 64, OW = 64, N = 64, IC = 64, OC = 64;
		//int IH = 32, IW = 32, OH = 32, OW = 32, N = 32, IC = 256, OC = 256;
		//int IH = 16, IW = 16, OH = 16, OW = 16, N = 32, IC = 512, OC = 512;
		
		//int IH = 120, IW = 120, OH = 120, OW = 120, N = 32, IC = 64, OC = 64;
		//int IH = 80, IW = 80, OH = 80, OW = 80, N = 64, IC = 64, OC = 64;
		//int IH = 40, IW = 40, OH = 40, OW = 40, N = 64, IC = 128, OC = 128;
		//int IH = 20, IW = 20, OH = 20, OW = 20, N = 64, IC = 256, OC = 256;
		//int IH = 10, IW = 10, OH = 10, OW = 10, N = 64, IC = 512, OC = 512;

		//======[8 * 8]=======================================================
		//int IH = 72, IW = 72, OH = 72, OW = 72, N = 64, IC = 64, OC = 64;
		//int IH = 36, IW = 36, OH = 36, OW = 36, N = 64, IC = 128, OC = 128;
		//int IH = 18, IW = 18, OH = 18, OW = 18, N = 64, IC = 256, OC = 256;
		//int IH = 9, IW = 9, OH = 9, OW = 9, N = 64, IC = 512, OC = 512;

		int IH = 64, IW = 64, OH = 64, OW = 64, N = 128, IC = 64, OC = 64;
		//int IH = 32, IW = 32, OH = 32, OW = 32, N = 128, IC = 128, OC = 128;
		//int IH = 16, IW = 16, OH = 16, OW = 16, N = 128, IC = 512, OC = 512;

		//int IH = 128, IW = 128, OH = 128, OW = 128, N = 32, IC = 64, OC = 64;
		//int IH = 64, IW = 64, OH = 64, OW = 64, N = 128, IC = 64, OC = 64;

		//int IH = 112, IW = 112, OH = 112, OW = 112, N = 32, IC = 64, OC = 64;
		//int IH = 56, IW = 56, OH = 56, OW = 56, N = 128, IC = 64, OC = 64;
		//int IH = 28, IW = 28, OH = 28, OW = 28, N = 128, IC = 128, OC = 128;

		//int IH = 144, IW = 144, OH = 144, OW = 144, N = 32, IC = 64, OC = 64;
		//int IH = 72, IW = 72, OH = 72, OW = 72, N = 128, IC = 64, OC = 64;
		//int IH = 36, IW = 36, OH = 36, OW = 36, N = 128, IC = 128, OC = 128;
		//int IH = 18, IW = 18, OH = 18, OW = 18, N = 128, IC = 256, OC = 256;

		//int IH = 40, IW = 40, OH = 40, OW = 40, N = 64, IC = 128, OC = 128;
		//int IH = 20, IW = 20, OH = 20, OW =20, N = 64, IC = 256, OC = 256;

		//======[9 * 9]=======================================================
		//int IH = 124, IW = 124, OH = 124, OW = 124, N = 32, IC = 64, OC = 64;
		//int IH = 60, IW = 60, OH = 60, OW = 60, N = 128, IC = 64, OC = 64;
		//int IH = 28, IW = 28, OH = 28, OW = 28, N = 128, IC = 128, OC = 128;

		//int IH = 128, IW = 128, OH = 128, OW = 128, N = 32, IC = 64, OC = 64;
		//int IH = 64, IW = 64, OH = 64, OW = 64, N = 128, IC = 64, OC = 64;
		//int IH = 32, IW = 32, OH = 32, OW = 32, N = 128, IC = 128, OC = 128;
		//int IH = 16, IW = 16, OH = 16, OW = 16, N = 128, IC = 256, OC = 256;
		//int IH = 8, IW = 8, OH = 8, OW = 8, N = 128, IC = 512, OC = 512;

		//int IH = 48, IW = 48, OH = 48, OW = 48, N = 64, IC = 64, OC = 64;

		if (FH == 8 && FW == 8) OH -= 1, OW -= 1;
		if (FH == 6 && FW == 6) OH -= 1, OW -= 1;
		if (FH == 4 && FW == 4) OH -= 1, OW -= 1;
		if (FH == 2 && FW == 2) OH -= 1, OW -= 1;

		//IH += 1; IW += 1; OH += 1; OW += 1;
		//IH += 2; IW += 2; OH += 2; OW += 2;
		//IH += 4; IW += 4; OH += 4; OW += 4;
		//IH -= 2; IW -= 2; OH -= 2; OW -= 2;

		//testCorrect<4>(IH/2, IW/2, OH/2, OW/2, FH, FW, N/4, IC, OC, sh, sw, ph, pw);//3*3*4 = 9*4=36
		//testCorrect<4>(IH/2, IW/2, OH/2, OW/2, FH, FW, N/4, IC/2, OC/2, sh, sw, ph, pw);//3*3*4 = 9*4=36
		//testCorrect<4>(IH/2, IW/2, OH/2, OW/2, FH, FW, N/2, IC/2, OC/2, sh, sw, ph, pw);//3*3*4 = 9*4=36
		//testCorrect<4>(IH, IW, OH, OW, FH, FW, N/4, IC/2, OC/2, sh, sw, ph, pw);//3*3*4 = 9*4=36

		testCorrect<4>(IH, IW, OH, OW, FH, FW, N/4, IC, OC/4, sh, sw, ph, pw);//3*3*4 = 9*4=36
		testSpeed<4>(1, IH, IW, OH, OW, FH, FW, N, IC, OC, sh, sw, ph, pw);//3*3*2 = 9*2
		testSpeed<4>(1000, IH, IW, OH, OW, FH, FW, N, IC, OC, sh, sw, ph, pw);//3*3*2 = 9*2
		//testSpeed<4>(1000, IH*2, IW*2, OH*2, OW*2, FH, FW, N, IC, OC, sh, sw, ph, pw);//3*3*2 = 9*2
		//testSpeed<4>(1000, IH, IW, OH, OW, FH, FW, N, IC, OC, sh, sw, ph, pw);//3*3*2 = 9*2
		//testSpeed<4>(200, IH, IW, OH, OW, FH, FW, N, IC*2, OC*2, sh, sw, ph, pw);//3*3*2 = 9*2
	}

	//======[Compress Area]===========================================
	{
		//int FH = 3, FW = 3, ph = 1, pw = 1;//4*4*8 = 32*4 = 128
		//int FH = 1, FW = 1, ph = 0, pw = 0;//4*4*8 = 32*4 = 128
		//int sh = 1, sw = 1;
		//int IH = 64, IW = 64, OH = 64, OW = 64, N = 32, IC = 16, OC = 256;
		//int IH = 32, IW = 32, OH = 32, OW = 32, N = 32, IC = 16, OC = 256;

		//testCorrect<4>(IH/2, IW/2, OH/2, OW/2, FH, FW, N/2, IC, OC, sh, sw, ph, pw);//3*3*4 = 9*4=36
		//testCorrect<4>(IH, IW, OH, OW, FH, FW, N/4, IC, OC, sh, sw, ph, pw);//3*3*4 = 9*4=36
		//testSpeed<4>(1000, IH, IW, OH, OW, FH, FW, N, IC, OC, sh, sw, ph, pw);//3*3*2 = 9*2
	}
}

main() {test();}

#endif//complie-area>>>>------------------------------------------------------------

#endif