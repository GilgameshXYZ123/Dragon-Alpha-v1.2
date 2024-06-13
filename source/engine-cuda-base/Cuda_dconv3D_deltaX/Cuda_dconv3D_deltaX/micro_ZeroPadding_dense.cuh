#pragma once

#ifndef MICRO_ZERO_PADDING_DENSE_H
#define MICRO_ZERO_PADDING_DENSE_H

#define GET_GN_ZeroPadding(IC) (IC) // GN = IC
#define GET_GM_ZeroPadding(N, IH, IW) ((N)*(IH)*(IW)) // GM = N  * IH * IW
#define GET_GK_ZeroPadding(OC, FH, FW) ((OC)*(FH)*(FW)) // GK = OC * FH * FW;


//====Improvement of j(n, ih, iw)(4x kernel)==================
//in k88: j % 8 == 0
//when (IH, IW) % 4 == 0
//(1) iw = j % IW = (j0 + i) % IW = (8x + i) % (4y) 
//So: iw0 = iw1 - 1 = iw2 - 2 = iw3 - 3
//So: iw4 = iw5 - 1 = iw6 - 2 = iw7 - 3
//(2) n = j / (IH * IW) = (j0 + i) / (IH * IW) 
//As: i belongs to (0, 7)
//So: n = (8*x + i) / (16y)
//So: ni = nj, i,j belongs to {0, 1, 2, 3, 4, 5, 6, 7}
//(3) ih = (j % (IH * IW)) / IW
//    ih = ((j0 + i) % (IH * IW)) / IW
//    ih = ((8x + i) % (16y)) / 4z
//As: (8x + i) % (16y) = 8*y + i
//So: ih = (8*y + i) / 4z
//So: ih0 = ih1 = ih2 = ih3 
//So: ih4 = ih5 = ih6 = ih7
//============================================================
#define get_n_ih_iw(j, n, ih, iw) \
	int n, ih, iw; {n = j / IH_IW; int jr = j - n * IH_IW; ih = jr / IW, iw = jr - ih * IW;}

#define get_n_ih_iw_Temp(j, n, ih, iw, IH_IW, IW) \
	int n, ih, iw; {n = j / IH_IW; int jr = j - n * IH_IW; ih = jr / IW, iw = jr - ih * IW;}

#define get_ih_iw_n(j, ih, iw, n) \
	int ih, iw, n; {ih = j / IW_N; int jr = j - ih * IW_N; iw = jr / N, n = jr - iw * N; }

#define get_ih_iw_n_Temp(j, ih, iw, n, IW_N, N) \
	int ih, iw, n; {ih = j / IW_N; int jr = j - ih * IW_N; iw = jr / N, n = jr - iw * N; }


#define load4d(V, A, w, z, y, x, Sz, Sy, Sx) \
	{if (((y)<0)||((y)>=Sy)||((z)<0)||((z)>= Sz)) (V) = 0;\
	 else (V) = get4d(A, w, z, y, x, Sz, Sy, Sx); }

#define load4d_S(V, A, w_Sz, z, y, x, Sz, Sy, Sx) \
	{if (((y)<0)||((y)>=Sy)||((z)<0)||((z)>= Sz)) (V) = 0;\
	 else (V) = get4d_S(A, w_Sz, z, y, x, Sy, Sx); }

#define load4d_check(V, ih, iw, value) \
	{if (((ih) < 0)||((ih) >= IH) || ((iw) < 0)||((iw) >= IW)) (V) = 0.0f; else (V) = (value);}


#define get_dY_oc_fh_fw_W3(k, oc, fh, fw) \
	oc = k / 9; k -= oc * 9;\
	fh = k / 3; fw = k - fh * 3;

//to compute the corresponding index of deltaY
//GK order(OC, FH, FW)
#define get_dY_oc_fh_fw(k, oc, fh, fw) \
	oc = k / FH_FW; k -= oc * FH_FW;\
	fh = k / FW; fw = k - fh * FW;

#define get_dY_oc_fh_fw_W2pow(k, oc, fh, fw) \
	oc = k >> LFH_FW; k &= FH_FW_m1;\
	fh = k >> LFW; fw = k & FW_m1;

#define get_fh_fw_oc(k, fh, fw, oc) \
	fh = k / FW_OC; k -= fh * FW_OC;\
	fw = k / OC; oc = k - fw * OC;

#define get_fh_fw_oc_OC2pow(k, fh, fw, oc) \
	fh = k / FW_OC; k -= fh * FW_OC;\
	fw = k >> LOC; oc = k & ((1 << LOC) - 1);


//to comoute the corresponding index of W
//Wr_fh_fw -> W[oc, FH - 1 - fh, FW - 1 -fw, ic]
// = (FH - 1 - W_fh)*FW + (FW - 1 - W_fw)
// = (FH*FW - 1) - (W_fh*FW + fw) = (FH_FW - 1) - W_k
#define get_W_oc_fh_fw(k, oc, r_fh_fw) \
	oc = k / FH_FW; k -= oc * FH_FW;\
	r_fh_fw = FH_FW - 1 - k;

#define get_W_oc_fh_fw_W2pow(k, oc, r_fh_fw) \
	oc = k >> LFH_FW; k &= FH_FW_m1;\
	r_fh_fw = FH_FW_m1 - k;


#define load4d_s1(V, n, oh, ow, oc) \
	if (oh < 0 || oh >= OH || ow < 0 || ow >= OW) (V) = 0.0f;\
	else (V) = get4d(deltaY, n, oh, ow, oc, OH, OW, OC);  


#ifndef S1_LOAD_YS4
#define S1_LOAD_YS4

__device__ __forceinline__ float4 S1_SaveYs4(const float* __restrict__ deltaY,
	int Y_fh, int Y_fw, int yoffset, int OH, int OW,
	int Y0, int tih0, int tiw0,
	int Y1, int tih1, int tiw1,
	int Y2, int tih2, int tiw2,
	int Y3, int tih3, int tiw3)
{
	OH -= Y_fh; OW -= Y_fw;
	bool ly0 = (tih0 >= -Y_fh) && (tih0 < OH) && (tiw0 >= -Y_fw) && (tiw0 < OW);
	bool ly1 = (tih1 >= -Y_fh) && (tih1 < OH) && (tiw1 >= -Y_fw) && (tiw1 < OW);
	bool ly2 = (tih2 >= -Y_fh) && (tih2 < OH) && (tiw2 >= -Y_fw) && (tiw2 < OW);
	bool ly3 = (tih3 >= -Y_fh) && (tih3 < OH) && (tiw3 >= -Y_fw) && (tiw3 < OW);

	float4 x;
	x.x = (ly0 ? deltaY[Y0 + yoffset] : 0);
	x.y = (ly1 ? deltaY[Y1 + yoffset] : 0);
	x.z = (ly2 ? deltaY[Y2 + yoffset] : 0);
	x.w = (ly3 ? deltaY[Y3 + yoffset] : 0);
	return x;
}

__device__ __forceinline__ float4 S1_SaveYs4x(const float* __restrict__ deltaY,
	int Y_fh, int Y_fw, int yoffset, int OH, int OW, int OC,
	int tih0, int tiw0, int tiw1, int tiw2, int tiw3)
{
	bool ly = (tih0 >= -Y_fh) && (tih0 < OH - Y_fh);
	bool ly0 = ly && (tiw0 >= -Y_fw) && (tiw0 < OW - Y_fw);
	bool ly1 = ly && (tiw1 >= -Y_fw) && (tiw1 < OW - Y_fw);
	bool ly2 = ly && (tiw2 >= -Y_fw) && (tiw2 < OW - Y_fw);
	bool ly3 = ly && (tiw3 >= -Y_fw) && (tiw3 < OW - Y_fw);

	float4 x;
	x.x = (ly0 ? deltaY[yoffset - OC] : 0);//Y0
	x.y = (ly1 ? deltaY[yoffset] : 0);//Y1
	x.z = (ly2 ? deltaY[yoffset + OC] : 0);//Y2
	x.w = (ly3 ? deltaY[yoffset + (OC << 1)] : 0);//Y3
	return x;
}

#endif


#ifndef S1_LOAD_YS4_TEXTURE
#define S1_LOAD_YS4_TEXTURE

__device__ __forceinline__ float4 S1_SaveYs4_tex(cudaTextureObject_t deltaY,
	int Y_fh, int Y_fw, int yoffset, int OH, int OW,
	int Y0, int tih0, int tiw0,
	int Y1, int tih1, int tiw1,
	int Y2, int tih2, int tiw2,
	int Y3, int tih3, int tiw3)
{
	float4 x;
	x.x = tex1Dfetch<float>(deltaY, Y0 + yoffset);
	x.y = tex1Dfetch<float>(deltaY, Y1 + yoffset);
	x.z = tex1Dfetch<float>(deltaY, Y2 + yoffset);
	x.w = tex1Dfetch<float>(deltaY, Y3 + yoffset);

	bool ly0 = LOAD_Y(tih0, tiw0, Y_fh, Y_fw); zero_float(x.x, ly0, x.x);
	bool ly1 = LOAD_Y(tih1, tiw1, Y_fh, Y_fw); zero_float(x.y, ly1, x.y);
	bool ly2 = LOAD_Y(tih2, tiw2, Y_fh, Y_fw); zero_float(x.z, ly2, x.z);
	bool ly3 = LOAD_Y(tih3, tiw3, Y_fh, Y_fw); zero_float(x.w, ly3, x.w);
	return x;
}

__device__ __forceinline__ float4 S1_SaveYs4x_tex(cudaTextureObject_t deltaY,
	int Y_fh, int Y_fw, int yoffset, int OH, int OW, int OC,
	int tih0, int tiw0, int tiw1, int tiw2, int tiw3)
{
	float4 x;
	x.x = tex1Dfetch<float>(deltaY, yoffset - OC);
	x.y = tex1Dfetch<float>(deltaY, yoffset);
	x.z = tex1Dfetch<float>(deltaY, yoffset + OC);
	x.w = tex1Dfetch<float>(deltaY, yoffset + (OC << 1));

	bool ly = (tih0 >= -Y_fh) && (tih0 < OH - Y_fh);
	bool ly0 = ly && (tiw0 >= -Y_fw) && (tiw0 < OW - Y_fw); zero_float(x.x, ly0, x.x);//Y0
	bool ly1 = ly && (tiw1 >= -Y_fw) && (tiw1 < OW - Y_fw); zero_float(x.y, ly1, x.y);//Y1
	bool ly2 = ly && (tiw2 >= -Y_fw) && (tiw2 < OW - Y_fw); zero_float(x.z, ly2, x.z);//Y2
	bool ly3 = ly && (tiw3 >= -Y_fw) && (tiw3 < OW - Y_fw); zero_float(x.w, ly3, x.w);//Y3
	return x;
}

#endif

#endif