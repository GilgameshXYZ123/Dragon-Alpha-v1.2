#pragma once

#ifndef MICRO_GEMMV2_H
#define MICRO_GEMMV2_H

#define CAN_V2_W3P1 ((FH == 3) && (FW == 3) && (ph == 1) && (pw == 1) && (IH > 1) && (IW > 1))
#define CAN_V2_W4P1 ((FH == 4) && (FW == 4) && (ph == 1) && (pw == 1) && (IH > 2) && (IW > 2))
#define CAN_V2_W5P2 ((FH == 5) && (FW == 5) && (ph == 2) && (pw == 2) && (IH > 2) && (IW > 2))

//[1] Ph = 2ph = (OH - 1)*sh - IH + FH;
//[2] Q = ((Ph + IH) * (Pw + IW)) / (IH * IW)
//[3] Ph + IH = (OH - 1)*sh + FH
//    Pw + IW = (OW - 1)*sw + FW
//	  Q = 1.0 * (((OH - 1)*sh + FH) * ((OW - 1)*sw + FW)) / (IH * IW)
#define PADDING_SCALE_UP(IH, IW, OH, OW, FH, FW, sh, sw) \
	(1.0 * (((OH - 1)*sh + FH) * ((OW - 1)*sw + FW)) / (IH * IW))

//V1.GMr = V2.Nr * OH * OW
//As: GM = V1.GMr + V1.j_index = (n_index + V2.Nr) * OH * OW = N * OH * OW
#define V2_TO_V1(FH, FW, OH, OW, N, IC, n_index)\
		const int OH_OW = OH * OW;\
		int GM = N * OH_OW, GK = FH * FW * IC;\
		int j_index = n_index * OH_OW;

#endif