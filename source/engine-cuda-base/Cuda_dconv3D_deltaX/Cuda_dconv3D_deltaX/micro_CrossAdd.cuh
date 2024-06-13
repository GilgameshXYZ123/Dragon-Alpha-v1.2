#pragma once

#ifndef MICRO_CROSS_ADD_H
#define MICRO_CROSS_ADD_H

__device__ __constant__ char XIDX_W3[] = {
	0, 1,  2,
	4, 5,  6,
	8, 9, 10 };

#define GET_GN_CrossAdd(OC) (OC)
#define GET_GM_CrossAdd(N, OH, OW) ((N)*(OH)*(OW))
#define GET_GK_CrossAdd(FH, FW, IC) ((FH)*(FW)*(IC))

#define get_n_oh_ow(j, n, oh, ow) \
	int n, oh, ow; {n = j / OH_OW; int jr = j - n * OH_OW; oh = jr / OW, ow = jr - oh * OW;}

#define getX_fh_fw(k, fh, fw) {fh = k / FW_IC; k -= fh * FW_IC; fw = k / IC;}
#define getX_fh_fw_ic2pow(k, fh, fw) {fh = k / FW_IC; k -= fh * FW_IC; fw = k >> LIC;}

#define CASM4(a, b) ((a).x*(b).x + (a).y*(b).y + (a).z*(b).z + (a).w *(b).w) //CrossAdd_Sum4
#define CASM2(a, b) ((a).x*(b).x + (a).y*(b).y) //CrossAdd_Sum2

#define WRITE_X(ih, iw, fh, fw) \
	((ih >= - fh)&&(ih < IH - fh) && (iw >= -fw)&&(iw < IW - fw))

#endif