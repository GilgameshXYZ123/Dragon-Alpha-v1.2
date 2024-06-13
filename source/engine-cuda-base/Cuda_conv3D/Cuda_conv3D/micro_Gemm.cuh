#pragma once

#ifndef MICRO_GEMM_H
#define MICRO_GEMM_H

#define GET_GN(OC) (OC)
#define GET_GM(N, OH, OW) ((N)*(OH)*(OW))
#define GET_GK(FH, FW, IC) ((FH)*(FW)*(IC))


#define GET_OUT_DIM(inDim, kernelDim, padding, step) (((inDim) + 2 * (padding) - (kernelDim)) / (step) + 1)


#define LOAD_X(ihs, iws, fh, fw) ((ihs >= -fh) && (ihs < IH - fh) && (iws >= -fw) && (iws < IW - fw))

#define load4d(V, A, w, z, y, x, Sz, Sy, Sx) \
	{(V) = ((((z)<0)||((z)>=Sz)||((y)<0)||((y)>= Sy))? 0.0f: A[(((w)*(Sz) + (z))*(Sy) + (y))*(Sx) + (x)]);}

#define load4d_tex(V, A, w, z, y, x, Sz, Sy, Sx) \
	{(V) = ((((z)<0)||((z)>=Sz)||((y)<0)||((y)>= Sy))? 0.0f: tex1Dfetch<float>(A, (((w)*(Sz) + (z))*(Sy) + (y))*(Sx) + (x)));}

#define load4d_IC2pow(V, A, w, z, y, x, Sz, Sy, LSx) \
	{(V) = ((z<0 || z>=Sz || y<0 || y>=Sy)? 0.0f: A[(((w*Sz + z)*Sy + y) << LSx) + x]);}

#define load4d_check(V, ih, iw, value) \
	{if (((ih) < 0)||((ih) >= IH) || ((iw) < 0)||((iw) >= IW)) (V) = 0.0f; else (V) = (value); }

#define simdAdd4(c, av, b) {(c).x = (av) + (b).x; (c).y = (av) + (b).y; (c).z = (av) + (b).z; (c).w = (av) + (b).w;}
#define vectorAdd4(c, a, b) { c.x = b.x + a.x; c.y = b.y + a.y; c.z = b.z + a.z; c.w = b.w + a.w;}


#define simdMM4(c, av, b) {(c).x += (av) * (b).x; (c).y += (av) * (b).y; (c).z += (av) * (b).z; (c).w += (av) * (b).w;}
#define simdMM2(c, av, b) {(c).x += (av) * (b).x; (c).y += (av) * (b).y;} 
#define simdMM4_xzyw(c, av, b) {(c).x += (av) * (b).x; (c).z += (av) * (b).y; (c).y += (av) * (b).z; (c).w += (av) * (b).w;}
#define rimdMM4(c, av, b) {(c).w += (av) * (b).w; (c).z += (av) * (b).z; (c).y += (av) * (b).y; (c).x += (av) * (b).x; }

//Shuffle Simd MatrixMultiply4
#define shuf_simdMM4(c, av, b, idx) {\
	(c).x += (av) * (b)[(3 * idx - 3) & 3];\
	(c).y += (av) * (b)[(3 * idx - 2) & 3];\
    (c).z += (av) * (b)[(3 * idx - 1) & 3];\
    (c).w += (av) * (b)[(3 * idx    ) & 3];}

#define Idx_simdMM4(c, av, b, i0, i1, i2, i3) {\
	(c).x += av * b[i0];\
	(c).y += av * b[i1];\
    (c).z += av * b[i2];\
    (c).w += av * b[i3];}


//prepare for GM = N*OH*OW
//=============Improvement of X[n, oh, ow, ic]=====================================
//j0 % 8 == 0
//(1) ni = ji / (OH * OW) = (j0 + i) / (16x), so: ni = nj
//(2) ihi = ((j0 + i)%(OH*OW)) / OW = (8*y + i)%(16*x) / 4*x
//So: ih0 = ih1 = ih2 = ih3, ih4 = ih5 = ih6 = ih7
//So: toh0 = toh1 = toh2 = toh3
//(3) iwi = (j0 + i)%OW = (8*x + i)%(4*y) 
//So: iw0 = iw1 - 1 = iw2 - 2 = iw3 - 3
//So: iw4 = iw5 - 1 = iw6 - 2 = iw7 - 3
//So: tow0 = tow1 - 1 = tow2 - 1 = tow3 - 1
//=============Improvement of X[n, oh, ow, ic]=====================================
#define get_n_oh_ow(j, n, oh, ow) \
	int n, oh, ow; {n = j / OH_OW; int jr = j - n * OH_OW; oh = jr / OW, ow = jr - oh * OW;}

#define get_n_oh_ow_Temp(j, n, oh, ow, OH_OW, OW) \
	int n, oh, ow; {n = j / OH_OW; int jr = j - n * OH_OW; oh = jr / OW, ow = jr - oh * OW;}

#define get_oh_ow_n(j, oh, ow, n) \
	int oh, ow, n; {oh = j / OW_N; int jr = j - oh * OW_N; ow = jr / N, n = jr - ow * N; }


//compute for GK = FH*FW*IC
//=============Improvement of X_k\W_k[fh, fw, ic]==================================
//[1] in k88
//X_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
//W_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
//X_k = ok*STEP + tx - ((tx >= STEP) << LB >> 1)
//W_k = ok*STEP + ty - ((ty >= STEP) << LB >> 1)
//Let: Ux = tx - ((tx >= STEP) << LB >> 1)
//Let: Uy = ty - ((ty >= STEP) << LB >> 1)
//X_k = ok*STEP + Ux
//W_k = ok*STEP + Uy
//[1.1] when LB = 4, IC % 8 == 0, we have: (tFH*IC) % 8 == 0, Ux, Uy belongs to [0, 7], STEP = 8
//X_fh = (ok*8*x + Ux) / tFW_IC = (ok*8*x + Ux)/8y
//W_fh = (ok*8*x + Uy) / tFW_IC = (ok*8*x + Ux)/8y
//So: X_fh = W_fh, when IC % 8 == 0
//[1.2] when LB = 3, IC % 4 == 0, we have: (tFH*IC) % 4 == 0, Ux, Uy belongs to [0, 3], STEP = 4
//X_fh = (ok*4*x + Ux) / tFW_IC = (ok*4*x + Ux)/4y
//W_fh = (ok*4*x + Uy) / tFW_IC = (ok*4*x + Ux)/4y
//So: X_fh = W_fh, when IC % 8 == 0
//
//[2] in k44
//X_k = ((ok << LB) + ty) >> 1 = ok*STEP + (ty >> 1)
//W_k = ((ok << LB) + tx) >> 1 = ok*STEP + (tx >> 1)
//So: when LB = 4\3, IC % [8\4] == 0, we have: X_fh = W_fh
//
//[3] in k84:
//W_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
//X_k = ((ok << LB) + ty) >> 1 = ok*STEP + (ty >> 1)
//So: when LB = 4\3, IC % [8\4] == 0, we have: X_fh = W_fh
//=============Improvement of X_k\W_k[fh, fw, ic]==================================
#define get_X_fh_fw_ic(k, fh, fw, ic) {\
	fh = k / FW_IC; k -= fh * FW_IC;\
	fw = k / IC; ic = k - fw * IC;}

//when IC is power of 2
#define get_X_fh_fw_ic_IC2pow(k, fh, fw, ic) {\
	fh = k / FW_IC; k -= fh * FW_IC;\
	fw = k >> LIC; ic = k & IC_m1;}

//when FW, IC is power of 2
#define get_X_fh_fw_ic_FW_IC2pow(k, fh, fw, ic) {\
	fh = k >> LFW_IC; k &= LFW_IC_m1;\
	fw = k >> LIC; ic = k & IC_m1;}


//======[for GK ordered by: <fh, fw, ic>]==================================================
#define get_X_fh_fw(k, fh, fw) { fh = k / FW_IC; k -= fh * FW_IC; fw = k / IC;}

#define get_X_fh_fw_IC2pow(k, fh, fw) { fh = k / FW_IC; k -= fh * FW_IC; fw = k >> LIC; }

#define get_X_fh_fw_FW_IC2pow(k, fh, fw) { fh = k >> LFW_IC; k &= LFW_IC_m1; fw = k >> LIC; }


//======[for GK ordered by: <ic, fh, fw>]==================================================
#define get_ic_fh_fw(k, ic, fh, fw) {\
	ic = k / FH_FW; k -= ic * FH_FW;\
	fh = k / FW; fw = k - fh * FW; }

#define get_ic_fh_fw_W2pow(k, ic, fh, fw) {\
	ic = k >> LFH_FW; k &= FH_FW_m1;\
	fh = k >> LFW; fw = k & FW_m1; }


#define shift_n_j(n, j) n *= OH_OW * OC; j *= OC;
#define shift_n_j_2pow(n, j) n = (n << LOH_OW) * OC; j *= OC;


#ifndef FUNCTION_SAVE_X4
#define FUNCTION_SAVE_X4

__device__ __forceinline__ float4 SaveX4(const float* __restrict__ X,
	int X_fh, int X_fw, int IH, int IW, int xoffset,
	int X0, int toh0, int tow0,
	int X1, int toh1, int tow1,
	int X2, int toh2, int tow2,
	int X3, int toh3, int tow3)
{
	IH -= X_fh; IW -= X_fw;
	bool lx0 = (toh0 >= -X_fh) && (toh0 < IH) && (tow0 >= -X_fw) && (tow0 < IW);
	bool lx1 = (toh1 >= -X_fh) && (toh1 < IH) && (tow1 >= -X_fw) && (tow1 < IW);
	bool lx2 = (toh2 >= -X_fh) && (toh2 < IH) && (tow2 >= -X_fw) && (tow2 < IW);
	bool lx3 = (toh3 >= -X_fh) && (toh3 < IH) && (tow3 >= -X_fw) && (tow3 < IW);

	float4 x;
	x.x = (lx0 ? X[X0 + xoffset] : 0);
	x.y = (lx1 ? X[X1 + xoffset] : 0);
	x.z = (lx2 ? X[X2 + xoffset] : 0);
	x.w = (lx3 ? X[X3 + xoffset] : 0);
	return x;
}

__device__ __forceinline__ float4 SaveX4x(const float* __restrict__ X,
	int X_fh, int X_fw, int IH, int IW, int xoffset, int sw_IC,
	int toh0, int tow0, int tow1, int tow2, int tow3)
{
	bool lx = (toh0 >= -X_fh) && (toh0 < IH - X_fh); IW -= X_fw;
	bool lx0 = lx && (tow0 >= -X_fw) && (tow0 < IW);
	bool lx1 = lx && (tow1 >= -X_fw) && (tow1 < IW);
	bool lx2 = lx && (tow2 >= -X_fw) && (tow2 < IW);
	bool lx3 = lx && (tow3 >= -X_fw) && (tow3 < IW);

	float4 x;
	x.x = (lx0 ? X[xoffset - sw_IC] : 0);
	x.y = (lx1 ? X[xoffset] : 0);
	x.z = (lx2 ? X[xoffset + sw_IC] : 0);
	x.w = (lx3 ? X[xoffset + (sw_IC << 1)] : 0);
	return x;
}

#endif


#ifndef FUNCTION_SAVE_X4_TEXTURE
#define FUNCTION_SAVE_X4_TEXTURE

__device__ __forceinline__ float4 SaveX4_tex(cudaTextureObject_t X,
	int X_fh, int X_fw, int IH, int IW, int xoffset,
	int X0, int toh0, int tow0,
	int X1, int toh1, int tow1,
	int X2, int toh2, int tow2,
	int X3, int toh3, int tow3)
{
	float4 x;
	x.x = tex1Dfetch<float>(X, X0 + xoffset);
	x.y = tex1Dfetch<float>(X, X1 + xoffset);
	x.z = tex1Dfetch<float>(X, X2 + xoffset);
	x.w = tex1Dfetch<float>(X, X3 + xoffset);

	IH -= X_fh; IW -= X_fw;
	bool lx0 = (toh0 >= -X_fh) && (toh0 < IH) && (tow0 >= -X_fw) && (tow0 < IW);
	bool lx1 = (toh1 >= -X_fh) && (toh1 < IH) && (tow1 >= -X_fw) && (tow1 < IW);
	bool lx2 = (toh2 >= -X_fh) && (toh2 < IH) && (tow2 >= -X_fw) && (tow2 < IW);
	bool lx3 = (toh3 >= -X_fh) && (toh3 < IH) && (tow3 >= -X_fw) && (tow3 < IW);
	zero_float(x.x, lx0, x.x);
	zero_float(x.y, lx1, x.y);
	zero_float(x.z, lx2, x.z);
	zero_float(x.w, lx3, x.w);
	return x;
}

__device__ __forceinline__ float4 SaveX4x_tex(cudaTextureObject_t X,
	int X_fh, int X_fw, int IH, int IW, int xoffset, int sw_IC,
	int toh0, int tow0, int tow1, int tow2, int tow3)
{
	float4 x;
	x.x = tex1Dfetch<float>(X, xoffset - sw_IC);
	x.y = tex1Dfetch<float>(X, xoffset);
	x.z = tex1Dfetch<float>(X, xoffset + sw_IC);
	x.w = tex1Dfetch<float>(X, xoffset + (sw_IC << 1));

	bool lx = (toh0 >= -X_fh) && (toh0 < IH - X_fh); IW -= X_fw;
	bool lx0 = lx && (tow0 >= -X_fw) && (tow0 < IW);
	bool lx1 = lx && (tow1 >= -X_fw) && (tow1 < IW);
	bool lx2 = lx && (tow2 >= -X_fw) && (tow2 < IW);
	bool lx3 = lx && (tow3 >= -X_fw) && (tow3 < IW);
	zero_float(x.x, lx0, x.x);
	zero_float(x.y, lx1, x.y);
	zero_float(x.z, lx2, x.z);
	zero_float(x.w, lx3, x.w);
	return x;
}

#endif


#ifndef FUNCTION_SAVE_X2
#define FUNCTION_SAVE_X2

__device__ __forceinline__ float2 SaveX2(const float* __restrict__ X,
	int X_fh, int X_fw, int IH, int IW, int xoffset,
	int X0, int toh0, int tow0,
	int X1, int toh1, int tow1)
{
	bool lx0 = (toh0 >= -X_fh) && (toh0 < IH - X_fh) && (tow0 >= -X_fw) && (tow0 < IW - X_fw);
	bool lx1 = (toh1 >= -X_fh) && (toh1 < IH - X_fh) && (tow1 >= -X_fw) && (tow1 < IW - X_fw);

	float2 x;
	x.x = (lx0 ? X[X0 + xoffset] : 0);
	x.y = (lx1 ? X[X1 + xoffset] : 0);
	return x;
}

#endif


#endif