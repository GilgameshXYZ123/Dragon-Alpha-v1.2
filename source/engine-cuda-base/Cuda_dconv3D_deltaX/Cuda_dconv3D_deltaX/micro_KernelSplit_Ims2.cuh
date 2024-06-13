#pragma once

//IH % sh == 0, IW % sw == 0, sh = sw = 2
#ifndef MICRO_KERNEL_SPLIT_INPUT_MOD_STEP2_H
#define MICRO_KERNEL_SPLIT_INPUT_MOD_STEP2_H

//As: CFW = (FW - x + 1) / 2, x belongs to {0, 1}
//LCFW = log2(CFW) = CFW >> 1
//when: FW = 2, CFW = (2 - 0 + 1)/2 = 1, CFW = (2 - 1 + 1)/2 = 1
//when: FW = 3, CFW = (3 - 0 + 1)/2 = 2, CFW = (3 - 1 + 1)/2 = 1
//when: FW = 4, CFW = (4 - 0 + 1)/2 = 2, CFW = (4 - 1 + 1)/2 = 2
#define Ims2_IS_CW_POWER2(FW) ((FW == 2) || (FW == 3) || (FW == 4))


#define Ims2_IH_slice(IH) ((IH) >> 1) //IH_slice = (IH + sh - 1) / sh
#define Ims2_IW_slice(IW) ((IW) >> 1) //IW_slice = (IW + sw - 1) / sw


#define Ims2_CFH(FH) ((FH + 1) >> 1) //((FH + sh - 1) / sh)
#define Ims2_CFW(FH) ((FW + 1) >> 1) // ((FW + sw - 1) / sw)


#define Ims2_GN(IC) (IC)
#define Ims2_GM(N, IH, IW) ((N)*(IH>>1)*(IW>>1)) //GM = N*IH_slice*IW_slice


//(CFH, CFW): the max (CFH, CFW)
#define Ims2_init(N, IH, IW, FH, FW, OC, IC) \
	int CFH = Ims2_CFH(FH);\
	int CFW = Ims2_CFW(FW);\
	int IH_slice = Ims2_IH_slice(IH);\
	int IW_slice = Ims2_IW_slice(IW);\
	int CWstride = Ims_CWstride(CFH, CFW, OC, IC);\
	int GN = IC; \
	int GM = N * IH_slice * IW_slice; \


#define Ims2_n_ih_iw(j, n, ih, iw) \
	int n, ih, iw; {\
	n = (j) / IH_IW_slice; int jr = (j) - n*IH_IW_slice;\
	ih = jr / IW_slice, iw = jr - ih*IW_slice;\
	ih = (ih << 1) + ihs; iw = (iw << 1) + iws; }

#define Ims2_ih_iw_n(j, ih, iw, n) \
	int ih, iw, n; {\
	ih = (j) / IW_slice_N; int jr = (j) - ih * IW_slice_N;\
	iw = jr / N; n = jr - iw * N;\
	ih = (ih << 1) + ihs; iw = (iw << 1) + iws; }


//(oc, fhr, fwr)============================================================
//<1> W_k & CFH_CFW_m1 = W_fhr*CFW + W_fwr
//<2> W_oc = W_k >> LCFH_LFW

#define Ims_oc_fhr_fwr(k, oc, fhr, fwr) int fhr, fwr, oc;\
	{ oc = k / CFH_CFW; k -= oc*CFH_CFW; fhr = k / CFW; fwr = k - fhr*CFW; }

#define Ims_oc_fhr_fwr_CW2pow(k, oc, fhr, fwr) int fhr, fwr, oc;\
	{ oc = k >> LCFH_CFW; k &= CFH_CFW_m1; fhr = k >> LCFW; fwr = k & opw; }

#define Ims_oc_CW2pow(k, oc) int fhr, fwr, oc;\
	{ oc = k >> LCFH_CFW; k &= CFH_CFW_m1;  }

//LCFH_CFW = 1 + 1 = 2, CFH_CFW_m1 = 2*2 - 1 = 3
#define Ims_oc_fhr_fwr_W3(k, oc, fhr, fwr) int fhr, fwr, oc;\
	{ oc = k >> 2; k &= 3; fhr = k >> 1; fwr = k & 1; }
//==========================================================================

#endif
