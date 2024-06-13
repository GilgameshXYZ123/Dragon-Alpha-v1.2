#pragma once

#ifndef MICRO_ZERO_PADDING_V2_DENSE_H
#define MICRO_ZERO_PADDING_V2_DENSE_H

#define CAN_s1_V2_W3P1 ((FH == 3) && (FW == 3) && (ph == 1) && (pw == 1) && (OH > 1) && (OW > 1))
#define CAN_s1_V2_W5P2 ((FH == 5) && (FW == 5) && (ph == 2) && (pw == 2) && (OH > 2) && (OW > 2)) 


//(OH - FH + 2oph) + 1 = IH
//[1] OH + Oph = 2oph = IH - 1 + FH
//[2] OW + Opw = 2opw = IW - 1 + FW
//[3] Q = ((OH + Oph) * (OW + Opw)) / (OH * OW)
//    Q = 1.0 * ((IH - 1 + FH) * (IW - 1 + FW)) / (IH * IW)
//(IH, IW) = 16: Qs1 = 1.26562
//(IH, IW) =  8: Qs1 = 1.5625
#define s1_PADDING_SCALE_UP(IH, IW, OH, OW, FH, FW) \
	(1.0 * ((IH - 1 + FH) * (IW - 1 + FW)) / (OH * OW))


//V1.GMr = V2.Nr * IH * IW
//As: GM = V1.GMr + V1.j_index = (n_index + V2.Nr) * IH * IW = N * IH * IW
#define s1_V2_TO_V1(FH, FW, IH, IW, N, OC, n_index) \
	const int IH_IW = IH * IW;\
	int GM = N * IH_IW, GK = FH * FW * OC;\
	int j_index = n_index * IH_IW;

#endif