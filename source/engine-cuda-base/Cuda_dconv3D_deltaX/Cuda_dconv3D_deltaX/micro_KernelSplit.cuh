#pragma once

#ifndef MICRO_KERNEL_SPLIT_H
#define MICRO_KERNEL_SPLIT_H

#define KS_IH_slice(IH, sh) ((IH + sh - 1)/(sh)) //IH_slice = (IH + sh - 1) / sh
#define KS_IW_slice(IW, sw) ((IW + sw - 1)/(sw)) //IW_slice = (IW + sw - 1) / sw
#define KS_CWstride(CFH, CFW, OC, IC) (CFH * CFW * OC * IC)


#define KS_GN(IC) (IC)
#define KS_GM(N, IH, IW, sh, sw) ((N)*(IH/sh)*(IW/sw)) //GM = N*IH_slice*IW_slice


#define KS_CFH(FH, sh) ((FH + sh - 1) / sh)
#define KS_CFW(FW, sw) ((FW + sw - 1) / sw)


//(CFH, CFW): the max (CFH, CFW)
#define KS_init(N, IH, IW, FH, FW, OC, IC, sh, sw) \
	int CFH = KS_CFH(FH, sh);\
	int CFW = KS_CFW(FW, sw);\
	int IH_slice = KS_IH_slice(IH, sh);\
	int IW_slice = KS_IW_slice(IW, sw);\
	int CWstride = KS_CWstride(CFH, CFW, OC, IC);\
	int GN = IC; \
	int GM = N * IH_slice * IW_slice;\


#define WRT_X(ih, iw) ((ih < IH) && (iw < IW))


//for k88, k88_oc2pow
#ifndef KS_SAVE_YS4
#define KS_SAVE_YS4

__device__ __forceinline__ float4 KS_SaveYs4(const float* __restrict__ deltaY,
	int tohs0, int tows0, int Y_fhr, int Y_fwr, int OH, int OW,
	int yoffset, int Y0, int Y1, int Y2, int Y3)
{
	bool ly0 = (tohs0 >= -Y_fhr) && (tohs0 < OH - Y_fhr) &&
		(tows0 >= -Y_fwr) && (tows0 < OW - Y_fwr);
	if (!ly0) return  FLOAT_ZERO4;

	float4 y;
	y.x = deltaY[Y0 + yoffset];
	y.y = deltaY[Y1 + yoffset];
	y.z = deltaY[Y2 + yoffset];
	y.w = deltaY[Y3 + yoffset];
	return y;
}

__device__ __forceinline__ float4 KS_SaveYs4_tex(cudaTextureObject_t deltaY,
	int tohs0, int tows0, int Y_fhr, int Y_fwr, int OH, int OW,
	int yoffset, int Y0, int Y1, int Y2, int Y3)
{
	bool ly0 = (tohs0 >= -Y_fhr) && (tohs0 < OH - Y_fhr) &&
		(tows0 >= -Y_fwr) && (tows0 < OW - Y_fwr);

	float4 y;
	zero_float(y.x, ly0, tex1Dfetch<float>(deltaY, Y0 + yoffset));
	zero_float(y.y, ly0, tex1Dfetch<float>(deltaY, Y1 + yoffset));
	zero_float(y.z, ly0, tex1Dfetch<float>(deltaY, Y2 + yoffset));
	zero_float(y.w, ly0, tex1Dfetch<float>(deltaY, Y3 + yoffset));
	return y;
}

#endif


//for k44, k44_oc2pow
#ifndef KS_SAVE_YS2
#define KS_SAVE_YS2

__device__ __forceinline__ float2 KS_SaveYs2(const float* __restrict__ deltaY,
	int tohs0, int tows0, int Y_fhr, int Y_fwr, int OH, int OW,
	int yoffset, int Y0, int Y1)
{
	bool ly0 = (tohs0 >= -Y_fhr) && (tohs0 < OH - Y_fhr) &&
		(tows0 >= -Y_fwr) && (tows0 < OW - Y_fwr);
	if (!ly0) return  FLOAT_ZERO2;

	float2 y;
	y.x = deltaY[Y0 + yoffset];
	y.y = deltaY[Y1 + yoffset];
	return y;
}

__device__ __forceinline__ float2 KS_SaveYs2_tex(cudaTextureObject_t deltaY,
	int tohs0, int tows0, int Y_fhr, int Y_fwr, int OH, int OW,
	int yoffset, int Y0, int Y1)
{
	bool ly0 = (tohs0 >= -Y_fhr) && (tohs0 < OH - Y_fhr) &&
		(tows0 >= -Y_fwr) && (tows0 < OW - Y_fwr);

	float2 y;
	zero_float(y.x, ly0, tex1Dfetch<float>(deltaY, Y0 + yoffset));
	zero_float(y.y, ly0, tex1Dfetch<float>(deltaY, Y1 + yoffset));
	return y;
}

#endif

#endif