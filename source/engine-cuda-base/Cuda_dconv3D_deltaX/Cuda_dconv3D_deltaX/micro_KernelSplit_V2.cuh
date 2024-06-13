#pragma once

#ifndef MICRO_KERNEL_SPLIT_V2
#define MICRO_KERNEL_SPLIT_V2

#define Ims2_CAN_W5(FH, FW, OH, OW) ((FH == 5) && (FW == 5) && (OH >= 2) && (OW >= 2))


//(OHp - FH + 2oph) + 1 = IH
//(OH + 2oph - CFH)/1 + 1 = 2oph + OH - CFH + 1 = IH_slice
//[1] Oph + OH = IH_slice + CFH - 1
//[2] Opw = OW = IW_slice + CFW - 1
//Q = avg: (Oph + OH) * (Opw + OW) / (OH * OW)
#define Ims2_PADDING_SCALE_UP(Q, IH_slice, IW_slice, OH, OW, FH, FW) {\
	int OH0 = (IH_slice) + ((FH + 1) >> 1) - 1;\
	int OW0 = (IW_slice) + ((FW + 1) >> 1) - 1;\
	int OH1 = (IH_slice) + (FH >> 1) - 1;\
	int OW1 = (IW_slice) + (FW >> 1) - 1;\
	Q = 0.25 * ((OH0 + OH1) * (OW0 + OW1)) / (OH * OW); }


//V1.GMr = V2.Nr * IH_slice * IW_slice
//As: GM = V1.GMr + V1.j_index = (n_index + V2.Nr) * IH_slice * IW_slice
//       = N * IH_slice * IW_slice
//[OH = OW = 8]: Q = 1.0625
//[OH = OW = 4]: Q = 1.125 *
//[OH = OW = 2]: Q = 1.25
#define KS_V2_TO_V1(FH, FW, IH_slice, IW_slice, N, OC, n_index) \
	const int IH_IW_slice = IH_slice * IW_slice;\
	int GM = N * IH_IW_slice;\
	int j_index = n_index * IH_IW_slice;


//(CFH, CFW): the max (CFH, CFW)
#define V2_Ims2_init(N, IH, IW, FH, FW, OC, IC) \
	int CFH = Ims2_CFH(FH);\
	int CFW = Ims2_CFW(FW);\
	int IH_slice = Ims2_IH_slice(IH);\
	int IW_slice = Ims2_IW_slice(IW);\
	int CWstride = Ims_CWstride(CFH, CFW, OC, IC);\

#endif
