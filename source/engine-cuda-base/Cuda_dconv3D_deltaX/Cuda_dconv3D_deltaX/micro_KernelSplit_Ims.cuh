#pragma once

//IH % sh == 0, IW % sw == 0
#ifndef MICRO_KERNEL_SPLIT_INPUT_MOD_STEP_H
#define MICRO_KERNEL_SPLIT_INPUT_MOD_STEP_H

#define Ims_IH_slice(IH, sh) ((IH)/(sh)) //IH_slice = (IH + sh - 1) / sh
#define Ims_IW_slice(IW, sw) ((IW)/(sw)) //IW_slice = (IW + sw - 1) / sw
#define Ims_CWstride(CFH, CFW, OC, IC) (CFH * CFW * OC * IC)


#define Ims_GN(IC) (IC)
#define Ims_GM(N, IH, IW, sh, sw) ((N)*(IH/sh)*(IW/sw)) //GM = N*IH_slice*IW_slice


#define Ims_CFH(FH, sh) ((FH + sh - 1) / sh)
#define Ims_CFW(FW, sw) ((FW + sw - 1) / sw)


//(CFH, CFW): the max (CFH, CFW)
#define Ims_init(N, IH, IW, FH, FW, OC, IC, sh, sw) \
	int CFH = Ims_CFH(FH, sh);\
	int CFW = Ims_CFW(FW, sw);\
	int IH_slice = Ims_IH_slice(IH, sh);\
	int IW_slice = Ims_IW_slice(IW, sw);\
	int CWstride = Ims_CWstride(CFH, CFW, OC, IC);\
	int GN = IC; \
	int GM = N * IH_slice * IW_slice;


//======Impovement for Xoffset(n, oh, ow)======================================
//------[Part1]----------------------------------------------------------------
//when (IH_slice, IW_slice) > 8
//xoffset[i] = ((ni*IH + ihi*sh + ihs)*IW + iwi*sw + iws)*IC + ic
//xoffset[i] = ((ni*IH + ihi*sh)*IW + iwi*sw)*IC + ic + (ihs*IW + iws)*IC
//let: C1 = (ihs*IW + iws)*IC
//let: Ui = (ni*IH + ihi*sh)*IW + iwi*sw
//xoffset[i] = Ui*IC + ic + C1
//
//Ui = (ni*IH + ihi*sh)*IW + iwi*sw
//Ui = (ni*IH_slice*sh + ihi*sh)*IW_slice*sw + iwi*sw
//As: IH_slice = IH/sh, IW_slice = IW/sw
//Ui = ni*IH_slice*IW_slice*sh*sw + ihi*IW_slice*sh*sw + iwi*sw
//Ui = sh*sw*(ni*IH_slice*IW_slice + ihi*IW_slice) + iwi*sw
//As: ji = ni*IH_slice*IW_slice + ihi*IW_slice + iwi
//we have:  ni*IH_slice*IW_slice + ihi*IW_slice = ji - iwi
//Ui = sh*sw*(ji - iwi) + iwi*sw
//Ui = sh*sw*ji - sh*sw*iwi + iwi*sw
//Ui = sh*sw*ji + iwi*sw*(1 - sh)
//As: iwi = ji % IW_slice
//Ui = sh*sw*ji + (ji % IW_slice)*sw*(1 - sh)
//Ui = sw*{sh*ji + (1 - sh)* (ji % IW_slice) }
//
//xoffset[i] = Ui*IC + ic + C1
//xoffset[i] = sw*{sh*ji + (1 - sh)* (ji % IW_slice) }*IC + ic + (ihs*IW + iws)*IC
//
//xoffset[i] = sw*{sh*ji + (1 - sh)* (ji % IW_slice) }*IC + ic + (ihs*IW + iws)*IC
//let: deltaX += ic0 + (ihs*IW + iws)*IC
//let: IC = IC*sw
//we have: xoffset[i] = (sh*ji + (1 - sh)* (ji % IW_slice))*IC;
//
//let: alpha = sh*IC, beta = (1-sh)*IC
//xoffset[i] = alpha*ji + beta*(j0 % IW_slice)
//In conclution:
//(1) deltaX += ic0 + (ihs*IW + iws)*IC
//(2) IC = IC*sw
//(3) alpha = sh*IC, beta = (1-sh)*IC
//(4) xoffset[i] = alpha*ji + beta*(ji % IW_slice)
//
//especilla, when sh = sw = 2
//(1) deltaX += ic0 + (ihs*IW + iws)*IC
//(2) alpha = IC * 4 =  IC << 2
//(3) beta = -2 * IC = -IC << 1
//(4) xoffset[i] = alpha*ji + beta*(ji % IW_slice)
//
//------[Part2]----------------------------------------------------------------
//xoffset = ((n*IH + ih*sh + ihs)*IW + iw*sw + iws)*IC + ic
//xoffset0 = ((n0*IH + ih0*sh + ihs)*IW + iw0*sw + iws)*IC + ic0
//let: U0 = (n0*IH + (ih0*sh + ihs))*IW + (iw0*sw + iws)
//xoffseti = Ui*IC + ic0;
//U0 = n0*IH*IW + (ih0*sh + ihs)*IW + (iw0*sw + iws)
//U0 = n0*IH*IW + ih0*sh*IW + iw0*sw + (ihs*IW + iws)
//as: IH%sh == 0, IW%sw == 0
//let: C = (ihs*IW + iws)
//U0 = n0*IH_slice*IW_slice*sh*sw + ih0*IW_slice*sh*sw + iw0*sw + C
//U0 = sh*sw*(n0*IH_slice*IW_slice + ih0*IW_slice) + iw0*sw + C
//As: j0 = n0*IH_slice*IW_slice + ih0*IW_slice + iw0
//U0 = sh*sw*(j0 - iw0) + iw0*sw + C
//U0 = sh*sw*j0 - sh*sw*iw0 + sw*iw0 + C
//As: ji = j0 + i,
//Ui = sh*sw*(j0 + i) - sh*sw*iwi + sw*iwi + C
//Ui = (sh*sw*j0 + C) + sh*sw*i - sh*sw*iwi + sw*iwi
//Let: G = (sh*sw*j0 + C)
//Ui = G + sh*sw(i - iwi) + sw*iwi
//
//As: iwi = (j0 + i) % IW_slice, 
//in k88: j0 % 8 == 0
//[1]: when IW_slice % 8 == 0, we have: iwi = iw0 + i
//so: Ui = G + sh*sw*(i - iw0 - i) + sw*(iw0 + i) = 
//    Ui = (G - sh*sw*iw0 + sw*iw0) + sw*i
//so: Ui = U0 + sw*i
//so: xoffset[i] = xoffset[i] + swi*IC
//[2]: when (IW_slice, IH_slice) % 8 == 0
//(1) we have: Ui = U0 + sw*i
//(2) we have: (IH_slice * IW_slice) % 64 == 0
//As: ni = ji / (IH_slice*IW_slice) = (j0 + i)/(IH_slice*IW_slice), As: i belongs to [0, 7]
//So: ni = (8*x + i)/(8*8*y) = 8*x / (8*8*y)
//So: ni = nj, i,j belongs to [0, 7]
//(3) we have: ji = ni*IH_slice*IW_slice + ihi*IW_slice + iwi
//ihi = ((j0 + i) % (IH_slice*IW_slice)) / IW_slice
//ihi = (j0 % (IH_slice*IW_slice) + i) / IW_slice, let: (j0 % (IH_slice*IW_slice) = V
//ihi = (V + i) / IW_slic, As: V % 8 == 0
//So: ihi = ihj, i,j belongs to [0, 7]
//
//int k88: j0 % 8 == 0
//[1]: when IW_slice % 4 == 0 && IH_slice % 4 == 0
//(1) ni = ji / (IH_slice*IW_slice) = (j0 + i)/(IH_slice*IW_slice), As: i belongs to [0, 7]
//So: ni = (8*x + i) / (4*4*y) = 8*x/(16*y)
//So: ni = nj, i,j belongs to [0, 7]
//(2) ih0 = ih1 = ih2 = ih3, ih4 = ih5 = ih6 = ih7
//As: ihi = (V + i) / IW_slice, As: V % 8 == 0
//ihi = (8*x + i) / (4 * y)
//(3) iw0 = iw1 - 1 = iw2 - 2 = iw3 - 3
//    iw4 = iw5 - 1 = iw6 - 2 = iw7 - 3
//======Impovement for Xoffset(n, oh, ow)======================================
#define Ims_n_ih_iw(j, n, ih, iw) \
	int n, ih, iw; {\
	n = (j) / IH_IW_slice; int jr = (j) - n*IH_IW_slice;\
	ih = jr / IW_slice, iw = jr - ih*IW_slice;\
	ih = (ih * sh) + ihs; iw = (iw * sw) + iws; }

//======Impovement for Xoffset(oh, ow, n)======================================
//-----[Part1] in k88----------------------------------------------------------
//N % 4 == 0, j0 % 8 == 0
//(1) n = (j0 + i) % N = (8x + i) / 4y
//So: n0 = n1 - 1 = n2 - 2 = n3 - 3
//So: n4 = n5 - 1 = n6 - 2 = n7 - 3
//(2) ihi = (j0 + i) / (N * IW_slice)
//    ihi = (8x + i)/4y
//So: ih0 = ih1 = ih2 = ih3
//So: ih4 = ih5 = ih6 = ih7
//(3) iwi = ((j0 + i) % (N * IW_slice)) / N
//    iwi = ((8x + i) % 4y) / 4z, i belongs to [0, 7]
//    iwi = (4u + i) / 4z, i belongs to [0, 3]
//So: iw0 = iw1 = iw2 = iw3;
//So: iw4 = iw5 = iw6 = iw7;
//(1) n = (j0 + i) % N
//So: n0 = n1 - 1 = n2 - 2 = n3 - 3
//So: n4 = n5 - 1 = n6 - 2 = n7 - 3
//
//-----[Part2] in k44----------------------------------------------------------
//N % 4 == 0, j0 % 4 == 0
//(1) n = (j0 + i) % N = (4x + i) / 4y
//So: n0 = n1 - 1 = n2 - 2 = n3 - 3
//(2) ihi = (j0 + i) / (N * IW_slice)
//    ihi = (4x + i)/4y
//So: ih0 = ih1 = ih2 = ih3
//(3) iwi = ((j0 + i) % (N * IW_slice)) / N
//    iwi = ((4x + i) % 4y) / 4z, i belongs to [0, 7]
//    iwi = (4u + i) / 4z, i belongs to [0, 3]
//So: iw0 = iw1 = iw2 = iw3;
//======Impovement for Xoffset(n, oh, ow)======================================
#define Ims_ih_iw_n(j, ih, iw, n) \
	int n, ih, iw; {\
	ih = (j) / IW_slice_N; int jr = (j) - ih*IW_slice_N;\
	iw = jr / N; n = jr - iw * N;\
	ih = (ih * sh) + ihs; iw = (iw * sw) + iws; }


#define Ims_ldy(ohs, ows) \
	((ohs >= -dY_fhr) && (ohs < OH-dY_fhr) && (ows>=-dY_fwr) && (ows < OW-dY_fwr))

#define Ims_ldy_ows(ows) \
	((ows>=-dY_fwr) && (ows < OW-dY_fwr))


//(fhr, fwr, oc)
#define Ims_fhr_fwr(k, fhr, fwr) int fhr, fwr;\
	{fhr = k / CFW_OC; k -= fhr * CFW_OC; fwr = k / OC; }

#define Ims_fhr_fwr_oc2pow(k, fhr, fwr) int fhr, fwr;\
	{fhr = k / CFW_OC; k -= fhr * CFW_OC; fwr = (k >> LOC); }

#define Ims_fhr_fwr_oc_CFW2pow(k, fhr, fwr) int fhr, fwr;\
	{fhr = k >> LCFW_OC; k &= CFW_OC_m1; fwr = (k >> LOC); }

#define Ims_fhr_fwr_W3_oc2pow(k, fhr, fwr) int fhr, fwr;\
	{fhr = k >> (LOC + 1); k &= CFW_OC_m1; fwr = (k >> LOC); }


#ifndef IMS_SAVE_YS4
#define IMS_SAVE_YS4

__device__ __forceinline__ float4 Ims_SaveYs4(const float* __restrict__ deltaY,
	int yoffset, int Y_fhr, int Y_fwr, int OH, int OW,
	int Y0, int tohs0, int tows0,
	int Y1, int tohs1, int tows1,
	int Y2, int tohs2, int tows2,
	int Y3, int tohs3, int tows3)
{
	OH -= Y_fhr; OW -= Y_fwr;
	bool ly0 = (tohs0 >= -Y_fhr) && (tohs0 < OH) && (tows0 >= -Y_fwr) && (tows0 < OW);
	bool ly1 = (tohs1 >= -Y_fhr) && (tohs1 < OH) && (tows1 >= -Y_fwr) && (tows1 < OW);
	bool ly2 = (tohs2 >= -Y_fhr) && (tohs2 < OH) && (tows2 >= -Y_fwr) && (tows2 < OW);
	bool ly3 = (tohs3 >= -Y_fhr) && (tohs3 < OH) && (tows3 >= -Y_fwr) && (tows3 < OW);

	float4 y;
	y.x = (ly0 ? deltaY[Y0 + yoffset] : 0);
	y.y = (ly1 ? deltaY[Y1 + yoffset] : 0);
	y.z = (ly2 ? deltaY[Y2 + yoffset] : 0);
	y.w = (ly3 ? deltaY[Y3 + yoffset] : 0);
	return y;
}

__device__ __forceinline__ float4 Ims4x_SaveYs4(const float* __restrict__ deltaY,
	int yoffset, int Y_fhr, int Y_fwr, int OH_m_tohs0, int OW, int OC,
	int tohs0, int tows0, int tows1, int tows2, int tows3)
{
	bool ly = (tohs0 >= -Y_fhr) && (Y_fhr < OH_m_tohs0); OW -= Y_fwr;
	bool ly0 = ly && (tows0 >= -Y_fwr) && (tows0 < OW);
	bool ly1 = ly && (tows1 >= -Y_fwr) && (tows1 < OW);
	bool ly2 = ly && (tows2 >= -Y_fwr) && (tows2 < OW);
	bool ly3 = ly && (tows3 >= -Y_fwr) && (tows3 < OW);

	float4 y;
	y.x = (ly0 ? deltaY[yoffset - OC] : 0);
	y.y = (ly1 ? deltaY[yoffset] : 0);
	y.z = (ly2 ? deltaY[yoffset + OC] : 0);
	y.w = (ly3 ? deltaY[yoffset + (OC << 1)] : 0);
	return y;
}

#endif


#ifndef IMS_SAVE_YS4_TEXTURE
#define IMS_SAVE_YS4_TEXTURE

__device__ __forceinline__ float4 Ims4x_SaveYs4_tex(
	cudaTextureObject_t deltaY,
	int yoffset, int Y_fhr, int Y_fwr, int OH_m_tohs0, int OW, int OC,
	int tohs0, int tows0, int tows1, int tows2, int tows3)
{
	float4 y;
	y.x = tex1Dfetch<float>(deltaY, yoffset - OC);
	y.y = tex1Dfetch<float>(deltaY, yoffset);
	y.z = tex1Dfetch<float>(deltaY, yoffset + OC);
	y.w = tex1Dfetch<float>(deltaY, yoffset + (OC << 1));

	bool ly = (tohs0 >= -Y_fhr) && (Y_fhr < OH_m_tohs0); OW -= Y_fwr;
	bool ly0 = ly && (tows0 >= -Y_fwr) && (tows0 < OW);
	bool ly1 = ly && (tows1 >= -Y_fwr) && (tows1 < OW);
	bool ly2 = ly && (tows2 >= -Y_fwr) && (tows2 < OW);
	bool ly3 = ly && (tows3 >= -Y_fwr) && (tows3 < OW);
	zero_float(y.x, ly0, y.x);
	zero_float(y.y, ly1, y.y);
	zero_float(y.z, ly2, y.z);
	zero_float(y.w, ly3, y.w);
	return y;
}

__device__ __forceinline__ float4 Ims_SaveYs4_tex(
	cudaTextureObject_t deltaY, int yoffset,
	int Y_fhr, int Y_fwr, int OH, int OW,
	int Y0, int tohs0, int tows0,
	int Y1, int tohs1, int tows1,
	int Y2, int tohs2, int tows2,
	int Y3, int tohs3, int tows3)
{
	float4 y;
	y.x = tex1Dfetch<float>(deltaY, Y0 + yoffset);
	y.y = tex1Dfetch<float>(deltaY, Y1 + yoffset);
	y.z = tex1Dfetch<float>(deltaY, Y2 + yoffset);
	y.w = tex1Dfetch<float>(deltaY, Y3 + yoffset);

	OH -= Y_fhr; OW -= Y_fwr;
	bool ly0 = (tohs0 >= -Y_fhr) && (tohs0 < OH) && (tows0 >= -Y_fwr) && (tows0 < OW);
	bool ly1 = (tohs1 >= -Y_fhr) && (tohs1 < OH) && (tows1 >= -Y_fwr) && (tows1 < OW);
	bool ly2 = (tohs2 >= -Y_fhr) && (tohs2 < OH) && (tows2 >= -Y_fwr) && (tows2 < OW);
	bool ly3 = (tohs3 >= -Y_fhr) && (tohs3 < OH) && (tows3 >= -Y_fwr) && (tows3 < OW);
	zero_float(y.x, ly0, y.x);
	zero_float(y.y, ly1, y.y);
	zero_float(y.z, ly2, y.z);
	zero_float(y.w, ly3, y.w);
	return y;
}


#endif


//for k88
#ifndef IMS_LOAD_YS4
#define IMS_LOAD_YS4

__device__ __forceinline__ float4 Ims_loadYs4(const float* __restrict__ deltaY,
	int Y_k, int OH, int OW, int OC, int CFW_OC, int OW_OC,
	int Y0, int tohs0, int tows0,
	int Y1, int tohs1, int tows1,
	int Y2, int tohs2, int tows2,
	int Y3, int tohs3, int tows3)
{
	Ims_fhr_fwr(Y_k, Y_fhr, Y_fwr);
	int yoffset = Y_fhr * OW_OC + Y_k;
	bool ly0 = (tohs0 >= -Y_fhr) && (tohs0 < OH - Y_fhr) && (tows0 >= -Y_fwr) && (tows0 < OW - Y_fwr);
	bool ly1 = (tohs1 >= -Y_fhr) && (tohs1 < OH - Y_fhr) && (tows1 >= -Y_fwr) && (tows1 < OW - Y_fwr);
	bool ly2 = (tohs2 >= -Y_fhr) && (tohs2 < OH - Y_fhr) && (tows2 >= -Y_fwr) && (tows2 < OW - Y_fwr);
	bool ly3 = (tohs3 >= -Y_fhr) && (tohs3 < OH - Y_fhr) && (tows3 >= -Y_fwr) && (tows3 < OW - Y_fwr);

	float4 y;
	y.x = (ly0 ? deltaY[Y0 + yoffset] : 0);
	y.y = (ly1 ? deltaY[Y1 + yoffset] : 0);
	y.z = (ly2 ? deltaY[Y2 + yoffset] : 0);
	y.w = (ly3 ? deltaY[Y3 + yoffset] : 0);
	return y;
}

__device__ __forceinline__ float4 Ims4x_loadYs4(const float* __restrict__ deltaY,
	int Y_k, int OH_m_tohs0, int OW, int OC, int CFW_OC, int OW_OC,
	int Y1, int tohs0, int tows0,
	int tows1, int tows2, int tows3)
{
	Ims_fhr_fwr(Y_k, Y_fhr, Y_fwr);
	int yoffset = Y_fhr * OW_OC + Y_k;
	bool ly = (tohs0 >= -Y_fhr) && (Y_fhr < OH_m_tohs0);
	bool ly0 = ly && (tows0 >= -Y_fwr) && (tows0 < OW - Y_fwr);
	bool ly1 = ly && (tows1 >= -Y_fwr) && (tows1 < OW - Y_fwr);
	bool ly2 = ly && (tows2 >= -Y_fwr) && (tows2 < OW - Y_fwr);
	bool ly3 = ly && (tows3 >= -Y_fwr) && (tows3 < OW - Y_fwr);

	float4 y;
	y.x = (ly0 ? deltaY[Y1 - OC + yoffset] : 0);
	y.y = (ly1 ? deltaY[Y1 + yoffset] : 0);
	y.z = (ly2 ? deltaY[Y1 + OC + yoffset] : 0);
	y.w = (ly3 ? deltaY[Y1 + (OC << 1) + yoffset] : 0);
	return y;
}

#endif


//for k88_oc2pow
#ifndef IMS_LOAD_YS4_OC_2POW
#define IMS_LOAD_YS4_OC_2POW

__device__ __forceinline__ float4 Ims_loadYs4_oc2pow(
	const float* __restrict__ deltaY,
	int Y_k, int OH, int OW, int LOC, int CFW_OC,
	int Y0, int tohs0, int tows0,
	int Y1, int tohs1, int tows1,
	int Y2, int tohs2, int tows2,
	int Y3, int tohs3, int tows3)
{
	Ims_fhr_fwr_oc2pow(Y_k, Y_fhr, Y_fwr);
	int yoffset = (Y_fhr << LOC) * OW + Y_k;
	bool ly0 = (tohs0 >= -Y_fhr) && (tohs0 < OH - Y_fhr) && (tows0 >= -Y_fwr) && (tows0 < OW - Y_fwr);
	bool ly1 = (tohs1 >= -Y_fhr) && (tohs1 < OH - Y_fhr) && (tows1 >= -Y_fwr) && (tows1 < OW - Y_fwr);
	bool ly2 = (tohs2 >= -Y_fhr) && (tohs2 < OH - Y_fhr) && (tows2 >= -Y_fwr) && (tows2 < OW - Y_fwr);
	bool ly3 = (tohs3 >= -Y_fhr) && (tohs3 < OH - Y_fhr) && (tows3 >= -Y_fwr) && (tows3 < OW - Y_fwr);

	float4 y;
	y.x = (ly0 ? deltaY[Y0 + yoffset] : 0);
	y.y = (ly1 ? deltaY[Y1 + yoffset] : 0);
	y.z = (ly2 ? deltaY[Y2 + yoffset] : 0);
	y.w = (ly3 ? deltaY[Y3 + yoffset] : 0);
	return y;
}

__device__ __forceinline__ float4 Ims4x_loadYs4_oc2pow(
	const float* __restrict__ deltaY,
	int Y_k, int OH_m_tohs0, int OW, int LOC, int CFW_OC,
	int Y1, int tohs0, int tows0,
	int tows1, int tows2, int tows3)
{
	Ims_fhr_fwr_oc2pow(Y_k, Y_fhr, Y_fwr);
	int yoffset = (Y_fhr << LOC) * OW + Y_k;
	bool ly = (tohs0 >= -Y_fhr) && (Y_fhr < OH_m_tohs0);
	bool ly0 = ly && (tows0 >= -Y_fwr) && (tows0 < OW - Y_fwr);
	bool ly1 = ly && (tows1 >= -Y_fwr) && (tows1 < OW - Y_fwr);
	bool ly2 = ly && (tows2 >= -Y_fwr) && (tows2 < OW - Y_fwr);
	bool ly3 = ly && (tows3 >= -Y_fwr) && (tows3 < OW - Y_fwr);

	float4 y;
	y.x = (ly0 ? deltaY[Y1 - (1 << LOC) + yoffset] : 0);
	y.y = (ly1 ? deltaY[Y1 + yoffset] : 0);
	y.z = (ly2 ? deltaY[Y1 + (1 << LOC) + yoffset] : 0);
	y.w = (ly3 ? deltaY[Y1 + (2 << LOC) + yoffset] : 0);
	return y;
}

#endif


//for k88_oc_CFW2pow
#ifndef IMS_LOAD_YS4_OC_CFW_2POW
#define IMS_LOAD_YS4_OC_CFW_2POW

__device__ __forceinline__ float4 Ims_loadYs4_oc_CFW2pow(
	const float* __restrict__ deltaY,
	int Y_k, int OH, int OW, int LOC, int LCFW_OC, int CFW_OC_m1,
	int Y0, int tohs0, int tows0,
	int Y1, int tohs1, int tows1,
	int Y2, int tohs2, int tows2,
	int Y3, int tohs3, int tows3)
{
	Ims_fhr_fwr_oc_CFW2pow(Y_k, Y_fhr, Y_fwr);
	int yoffset = (Y_fhr << LOC) * OW + Y_k;
	bool ly0 = (tohs0 >= -Y_fhr) && (tohs0 < OH - Y_fhr) && (tows0 >= -Y_fwr) && (tows0 < OW - Y_fwr);
	bool ly1 = (tohs1 >= -Y_fhr) && (tohs1 < OH - Y_fhr) && (tows1 >= -Y_fwr) && (tows1 < OW - Y_fwr);
	bool ly2 = (tohs2 >= -Y_fhr) && (tohs2 < OH - Y_fhr) && (tows2 >= -Y_fwr) && (tows2 < OW - Y_fwr);
	bool ly3 = (tohs3 >= -Y_fhr) && (tohs3 < OH - Y_fhr) && (tows3 >= -Y_fwr) && (tows3 < OW - Y_fwr);

	float4 y;
	y.x = (ly0 ? deltaY[Y0 + yoffset] : 0);
	y.y = (ly1 ? deltaY[Y1 + yoffset] : 0);
	y.z = (ly2 ? deltaY[Y2 + yoffset] : 0);
	y.w = (ly3 ? deltaY[Y3 + yoffset] : 0);
	return y;
}


__device__ __forceinline__ float4 Ims4x_loadYs4_oc_CFW2pow(
	const float* __restrict__ deltaY,
	int Y_k, int OH_m_tohs0, int OW, int LOC, int LCFW_OC, int CFW_OC_m1,
	int Y1, int tohs0,
	int tows0, int tows1, int tows2, int tows3)
{
	Ims_fhr_fwr_oc_CFW2pow(Y_k, Y_fhr, Y_fwr);
	int yoffset = (Y_fhr << LOC) * OW + Y_k; OW -= Y_fwr;
	bool ly = (tohs0 >= -Y_fhr) && (Y_fhr < OH_m_tohs0);
	bool ly0 = ly && (tows0 >= -Y_fwr) && (tows0 < OW);
	bool ly1 = ly && (tows1 >= -Y_fwr) && (tows1 < OW);
	bool ly2 = ly && (tows2 >= -Y_fwr) && (tows2 < OW);
	bool ly3 = ly && (tows3 >= -Y_fwr) && (tows3 < OW);

	float4 y;
	y.x = (ly0 ? deltaY[Y1 - (1 << LOC) + yoffset] : 0);
	y.y = (ly1 ? deltaY[Y1 + yoffset] : 0);
	y.z = (ly2 ? deltaY[Y1 + (1 << LOC) + yoffset] : 0);
	y.w = (ly3 ? deltaY[Y1 + (2 << LOC) + yoffset] : 0);
	return y;
}

#endif


//for k88_oc_CFW2pow_tex
#ifndef IMS_LOAD_YS4_OC_CFW_2POW_TEXTURE
#define IMS_LOAD_YS4_OC_CFW_2POW_TEXTURE

__device__ __forceinline__ float4 Ims_loadYs4_oc_CFW2pow_tex(
	cudaTextureObject_t deltaY,
	int Y_k, int OH, int OW, int LOC, int LCFW_OC, int CFW_OC_m1,
	int Y0, int tohs0, int tows0,
	int Y1, int tohs1, int tows1,
	int Y2, int tohs2, int tows2,
	int Y3, int tohs3, int tows3)
{
	Ims_fhr_fwr_oc_CFW2pow(Y_k, Y_fhr, Y_fwr);
	int yoffset = (Y_fhr << LOC) * OW + Y_k;
	bool ly0 = (tohs0 >= -Y_fhr) && (tohs0 < OH - Y_fhr) && (tows0 >= -Y_fwr) && (tows0 < OW - Y_fwr);
	bool ly1 = (tohs1 >= -Y_fhr) && (tohs1 < OH - Y_fhr) && (tows1 >= -Y_fwr) && (tows1 < OW - Y_fwr);
	bool ly2 = (tohs2 >= -Y_fhr) && (tohs2 < OH - Y_fhr) && (tows2 >= -Y_fwr) && (tows2 < OW - Y_fwr);
	bool ly3 = (tohs3 >= -Y_fhr) && (tohs3 < OH - Y_fhr) && (tows3 >= -Y_fwr) && (tows3 < OW - Y_fwr);

	float4 y;
	zero_float(y.x, ly0, tex1Dfetch<float>(deltaY, Y0 + yoffset));
	zero_float(y.y, ly1, tex1Dfetch<float>(deltaY, Y1 + yoffset));
	zero_float(y.z, ly2, tex1Dfetch<float>(deltaY, Y2 + yoffset));
	zero_float(y.w, ly3, tex1Dfetch<float>(deltaY, Y3 + yoffset));
	return y;
}


__device__ __forceinline__ float4 Ims4x_loadYs4_oc_CFW2pow_tex(
	cudaTextureObject_t deltaY,
	int Y_k, int OH_m_tohs0, int OW, int LOC, int LCFW_OC, int CFW_OC_m1,
	int Y1, int tohs0,
	int tows0, int tows1, int tows2, int tows3)
{
	Ims_fhr_fwr_oc_CFW2pow(Y_k, Y_fhr, Y_fwr);
	int yoffset = (Y_fhr << LOC) * OW + Y_k; OW -= Y_fwr;
	bool ly = (tohs0 >= -Y_fhr) && (Y_fhr < OH_m_tohs0);
	bool ly0 = ly && (tows0 >= -Y_fwr) && (tows0 < OW);
	bool ly1 = ly && (tows1 >= -Y_fwr) && (tows1 < OW);
	bool ly2 = ly && (tows2 >= -Y_fwr) && (tows2 < OW);
	bool ly3 = ly && (tows3 >= -Y_fwr) && (tows3 < OW);

	float4 y;
	zero_float(y.x, ly0, tex1Dfetch<float>(deltaY, Y1 - (1 << LOC) + yoffset));
	zero_float(y.y, ly1, tex1Dfetch<float>(deltaY, Y1 + yoffset));
	zero_float(y.z, ly2, tex1Dfetch<float>(deltaY, Y1 + (1 << LOC) + yoffset));
	zero_float(y.w, ly3, tex1Dfetch<float>(deltaY, Y1 + (2 << LOC) + yoffset));
	return y;

	//Ims_fhr_fwr_oc_CFW2pow(Y_k, Y_fhr, Y_fwr);
	//float4 y; int yoffset = Y1 + (Y_fhr << LOC) * OW + Y_k; LOC = (1 << LOC);
	//y.x = tex1Dfetch<float>(deltaY, yoffset - LOC);
	//y.y = tex1Dfetch<float>(deltaY, yoffset);
	//y.z = tex1Dfetch<float>(deltaY, yoffset + LOC);
	//y.w = tex1Dfetch<float>(deltaY, yoffset + (LOC << 1));

	//OW -= Y_fwr;
	//bool ly = (tohs0 >= -Y_fhr) && (Y_fhr < OH_m_tohs0);
	//bool ly0 = ly && (tows0 >= -Y_fwr) && (tows0 < OW); zero_float(y.x, ly0, y.x);
	//bool ly1 = ly && (tows1 >= -Y_fwr) && (tows1 < OW); zero_float(y.y, ly1, y.y);
	//bool ly2 = ly && (tows2 >= -Y_fwr) && (tows2 < OW); zero_float(y.z, ly2, y.z);
	//bool ly3 = ly && (tows3 >= -Y_fwr) && (tows3 < OW); zero_float(y.w, ly3, y.w);
	//return y;
}

#endif


//for k88_tex
#ifndef IMS_LOAD_YS4_TEXTURE
#define IMS_LOAD_YS4_TEXTURE

__device__ __forceinline__ float4 Ims_loadYs4_tex(
	cudaTextureObject_t deltaY,
	int Y_k, int OH, int OW, int OC, int CFW_OC, int OW_OC,
	int Y0, int tohs0, int tows0,
	int Y1, int tohs1, int tows1,
	int Y2, int tohs2, int tows2,
	int Y3, int tohs3, int tows3)
{
	Ims_fhr_fwr(Y_k, Y_fhr, Y_fwr);
	int yoffset = Y_fhr * OW_OC + Y_k;
	bool ly0 = (tohs0 >= -Y_fhr) && (tohs0 < OH - Y_fhr) && (tows0 >= -Y_fwr) && (tows0 < OW - Y_fwr);
	bool ly1 = (tohs1 >= -Y_fhr) && (tohs1 < OH - Y_fhr) && (tows1 >= -Y_fwr) && (tows1 < OW - Y_fwr);
	bool ly2 = (tohs2 >= -Y_fhr) && (tohs2 < OH - Y_fhr) && (tows2 >= -Y_fwr) && (tows2 < OW - Y_fwr);
	bool ly3 = (tohs3 >= -Y_fhr) && (tohs3 < OH - Y_fhr) && (tows3 >= -Y_fwr) && (tows3 < OW - Y_fwr);

	float4 y;
	zero_float(y.x, ly0, tex1Dfetch<float>(deltaY, Y0 + yoffset));
	zero_float(y.y, ly1, tex1Dfetch<float>(deltaY, Y1 + yoffset));
	zero_float(y.z, ly2, tex1Dfetch<float>(deltaY, Y2 + yoffset));
	zero_float(y.w, ly3, tex1Dfetch<float>(deltaY, Y3 + yoffset));
	return y;
}

__device__ __forceinline__ float4 Ims4x_loadYs4_tex(
	cudaTextureObject_t deltaY,
	int Y_k, int OH_m_tohs0, int OW, int OC, int CFW_OC, int OW_OC,
	int Y1, int tohs0, int tows0,
	int tows1, int tows2, int tows3)
{
	Ims_fhr_fwr(Y_k, Y_fhr, Y_fwr);
	int yoffset = Y_fhr * OW_OC + Y_k;
	bool ly = (tohs0 >= -Y_fhr) && (Y_fhr < OH_m_tohs0);
	bool ly0 = ly && (tows0 >= -Y_fwr) && (tows0 < OW - Y_fwr);
	bool ly1 = ly && (tows1 >= -Y_fwr) && (tows1 < OW - Y_fwr);
	bool ly2 = ly && (tows2 >= -Y_fwr) && (tows2 < OW - Y_fwr);
	bool ly3 = ly && (tows3 >= -Y_fwr) && (tows3 < OW - Y_fwr);

	float4 y;
	zero_float(y.x, ly0, tex1Dfetch<float>(deltaY, Y1 - OC + yoffset));
	zero_float(y.y, ly1, tex1Dfetch<float>(deltaY, Y1 + yoffset));
	zero_float(y.z, ly2, tex1Dfetch<float>(deltaY, Y1 + OC + yoffset));
	zero_float(y.w, ly3, tex1Dfetch<float>(deltaY, Y1 + (OC << 1) + yoffset));
	return y;
}

#endif


//for k88_oc2pow_tex
#ifndef IMS_LOAD_YS4_OC_2POW_TEXTURE
#define IMS_LOAD_YS4_OC_2POW_TEXTURE

__device__ __forceinline__ float4 Ims_loadYs4_oc2pow_tex(
	cudaTextureObject_t deltaY,
	int Y_k, int OH, int OW, int LOC, int CFW_OC,
	int Y0, int tohs0, int tows0,
	int Y1, int tohs1, int tows1,
	int Y2, int tohs2, int tows2,
	int Y3, int tohs3, int tows3)
{
	Ims_fhr_fwr_oc2pow(Y_k, Y_fhr, Y_fwr);
	int yoffset = (Y_fhr << LOC) * OW + Y_k;
	bool ly0 = (tohs0 >= -Y_fhr) && (tohs0 < OH - Y_fhr) && (tows0 >= -Y_fwr) && (tows0 < OW - Y_fwr);
	bool ly1 = (tohs1 >= -Y_fhr) && (tohs1 < OH - Y_fhr) && (tows1 >= -Y_fwr) && (tows1 < OW - Y_fwr);
	bool ly2 = (tohs2 >= -Y_fhr) && (tohs2 < OH - Y_fhr) && (tows2 >= -Y_fwr) && (tows2 < OW - Y_fwr);
	bool ly3 = (tohs3 >= -Y_fhr) && (tohs3 < OH - Y_fhr) && (tows3 >= -Y_fwr) && (tows3 < OW - Y_fwr);

	float4 y;
	zero_float(y.x, ly0, tex1Dfetch<float>(deltaY, Y0 + yoffset));
	zero_float(y.y, ly1, tex1Dfetch<float>(deltaY, Y1 + yoffset));
	zero_float(y.z, ly2, tex1Dfetch<float>(deltaY, Y2 + yoffset));
	zero_float(y.w, ly3, tex1Dfetch<float>(deltaY, Y3 + yoffset));
	return y;
}

__device__ __forceinline__ float4 Ims4x_loadYs4_oc2pow_tex(
	cudaTextureObject_t deltaY,
	int Y_k, int OH_m_tohs0, int OW, int LOC, int CFW_OC,
	int Y1, int tohs0, int tows0,
	int tows1, int tows2, int tows3)
{
	Ims_fhr_fwr_oc2pow(Y_k, Y_fhr, Y_fwr);
	int yoffset = (Y_fhr << LOC) * OW + Y_k;
	bool ly = (tohs0 >= -Y_fhr) && (Y_fhr < OH_m_tohs0);
	bool ly0 = ly && (tows0 >= -Y_fwr) && (tows0 < OW - Y_fwr);
	bool ly1 = ly && (tows1 >= -Y_fwr) && (tows1 < OW - Y_fwr);
	bool ly2 = ly && (tows2 >= -Y_fwr) && (tows2 < OW - Y_fwr);
	bool ly3 = ly && (tows3 >= -Y_fwr) && (tows3 < OW - Y_fwr);

	float4 y;
	zero_float(y.x, ly0, tex1Dfetch<float>(deltaY, Y1 - (1 << LOC) + yoffset));
	zero_float(y.y, ly1, tex1Dfetch<float>(deltaY, Y1 + yoffset));
	zero_float(y.z, ly2, tex1Dfetch<float>(deltaY, Y1 + (1 << LOC) + yoffset));
	zero_float(y.w, ly3, tex1Dfetch<float>(deltaY, Y1 + (2 << LOC) + yoffset));
	return y;
}

#endif


//for k44, k44_oc2pow
#ifndef IMS_LOAD_YS2
#define IMS_LOAD_YS2

__device__ __forceinline__ float2 Ims_loadYs2(
	const float* __restrict__ deltaY,
	int Y_k, int OH, int OW, int OC, int CFW_OC, int OW_OC,
	int Y0, int tohs0, int tows0,
	int Y1, int tohs1, int tows1)
{
	Ims_fhr_fwr(Y_k, Y_fhr, Y_fwr);
	int yoffset = Y_fhr * OW_OC + Y_k;
	bool ly0 = (tohs0 >= -Y_fhr) && (tohs0 < OH - Y_fhr) && (tows0 >= -Y_fwr) && (tows0 < OW - Y_fwr);
	bool ly1 = (tohs1 >= -Y_fhr) && (tohs1 < OH - Y_fhr) && (tows1 >= -Y_fwr) && (tows1 < OW - Y_fwr);

	float2 y;
	y.x = (ly0 ? deltaY[Y0 + yoffset] : 0);
	y.y = (ly1 ? deltaY[Y1 + yoffset] : 0);
	return y;
}

__device__ __forceinline__ float2 Ims_loadYs2_oc2pow(
	const float* __restrict__ deltaY,
	int Y_k, int OH, int OW, int LOC, int CFW_OC,
	int Y0, int tohs0, int tows0,
	int Y1, int tohs1, int tows1)
{
	Ims_fhr_fwr_oc2pow(Y_k, Y_fhr, Y_fwr);
	int yoffset = (Y_fhr << LOC) * OW + Y_k;
	bool ly0 = (tohs0 >= -Y_fhr) && (tohs0 < OH - Y_fhr) && (tows0 >= -Y_fwr) && (tows0 < OW - Y_fwr);
	bool ly1 = (tohs1 >= -Y_fhr) && (tohs1 < OH - Y_fhr) && (tows1 >= -Y_fwr) && (tows1 < OW - Y_fwr);

	float2 y;
	y.x = (ly0 ? deltaY[Y0 + yoffset] : 0);
	y.y = (ly1 ? deltaY[Y1 + yoffset] : 0);
	return y;
}

#endif


//for k44_tex, k44_oc2pow
#ifndef IMS_LOAD_YS2_TEX
#define IMS_LOAD_YS2_TEX

__device__ __forceinline__ float2 Ims_loadYs2_tex(
	cudaTextureObject_t deltaY,
	int Y_k, int OH, int OW, int OC, int CFW_OC, int OW_OC,
	int Y0, int tohs0, int tows0,
	int Y1, int tohs1, int tows1)
{
	Ims_fhr_fwr(Y_k, Y_fhr, Y_fwr);
	int yoffset = Y_fhr * OW_OC + Y_k;
	bool ly0 = (tohs0 >= -Y_fhr) && (tohs0 < OH - Y_fhr) && (tows0 >= -Y_fwr) && (tows0 < OW - Y_fwr);
	bool ly1 = (tohs1 >= -Y_fhr) && (tohs1 < OH - Y_fhr) && (tows1 >= -Y_fwr) && (tows1 < OW - Y_fwr);

	float2 y;
	zero_float(y.x, ly0, tex1Dfetch<float>(deltaY, Y0 + yoffset));
	zero_float(y.y, ly1, tex1Dfetch<float>(deltaY, Y1 + yoffset));
	return y;
}

__device__ __forceinline__ float2 Ims_loadYs2_oc2pow_tex(
	cudaTextureObject_t deltaY,
	int Y_k, int OH, int OW, int LOC, int CFW_OC,
	int Y0, int tohs0, int tows0,
	int Y1, int tohs1, int tows1)
{
	Ims_fhr_fwr_oc2pow(Y_k, Y_fhr, Y_fwr);
	int yoffset = (Y_fhr << LOC) * OW + Y_k;
	bool ly0 = (tohs0 >= -Y_fhr) && (tohs0 < OH - Y_fhr) && (tows0 >= -Y_fwr) && (tows0 < OW - Y_fwr);
	bool ly1 = (tohs1 >= -Y_fhr) && (tohs1 < OH - Y_fhr) && (tows1 >= -Y_fwr) && (tows1 < OW - Y_fwr);

	float2 y;
	zero_float(y.x, ly0, tex1Dfetch<float>(deltaY, Y0 + yoffset));
	zero_float(y.y, ly1, tex1Dfetch<float>(deltaY, Y1 + yoffset));
	return y;
}

#endif

#endif
