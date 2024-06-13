#pragma once

#ifndef CONV_3D_WINOGRAD_2D_F22X33R_UTIL_H
#define CONV_3D_WINOGRAD_2D_F22X33R_UTIL_H

#define winnograd2D_f22x33_get_nhw(j, n, h, w) {\
	n = j / THW; int jr = j - n * THW;\
	int th = jr / TW, tw = jr - th * TW;\
	h = (th << 1); w = (tw << 1); }

__device__ __forceinline__ void Winograd2D_f22x33_transform_Y(const float* a, float *y) {
	float k[8];//k[2, 4], stride = 4
	k[0] = a[0] + a[4] + a[ 8]; k[4] = a[4] - a[ 8] - a[12];//k00 = a00 + a10 + a20; k10 = a10 - a20 - a30
	k[1] = a[1] + a[5] + a[ 9]; k[5] = a[5] - a[ 9] - a[13];//k01 = a01 + a11 + a21; k11 = a11 - a21 - a31;
	k[2] = a[2] + a[6] + a[10]; k[6] = a[6] - a[10] - a[14];//k02 = a02 + a12 + a22; k12 = a12 - a22 - a32;
	k[3] = a[3] + a[7] + a[11]; k[7] = a[7] - a[11] - a[15];//k03 = a03 + a13 + a23; k13 = a13 - a23 - a33;

	//a[2, 2] stride = 2
	y[0] = k[0] + k[1] + k[2]; y[1] = k[1] - k[2] - k[3];//a00 = k00 + k01 + k02; a01 = k01 - k02 - k03; 
	y[2] = k[4] + k[5] + k[6]; y[3] = k[5] - k[6] - k[7];//a10 = k10 + k11 + k12; a11 = k11 - k12 - k13;
}


__device__ __forceinline__ void Winograd2D_f22x33_transform_X_column(float* x) {
#pragma unroll
	for (int t = 0; t < 4; t++) {//for each column
		float x1 = x[4 + t], x2 = x[8 + t];
		x[     t] += -x2;          //x0t - x2t
		x[ 4 + t] +=  x2;          //x1t + x2t
		x[ 8 + t] += -x1;          //x2t - x1t
		x[12 + t] = x1 - x[12 + t];//x1t - x3t
	}
}

__device__ __forceinline__ void Winograd2D_f22x33_transform_X(float* x) {
#pragma unroll
	for (int t = 0; t < 4; t++) {//for each column
		float x1 = x[ 4 + t];//(1, t)
		float x2 = x[ 8 + t];//(2, t)
		x[t     ] = x[t] - x2;     //x0t - x2t
		x[4  + t] = x1 + x2;       //x1t + x2t
		x[8  + t] = x2 - x1;       //x2t - x1t
		x[12 + t] = x1 - x[12 + t];//x1t - x3t
	}

#pragma unroll
	for (int t = 0; t < 4; t++) {//for each row
		float h1 = x[(t << 2) + 1];//(t, 1)
		float h2 = x[(t << 2) + 2];//(t, 2)
		x[(t << 2)    ] = x[(t << 2)] - h2;    //ht0 - ht2
		x[(t << 2) + 1] = h1 + h2;             //ht1 + ht2
		x[(t << 2) + 2] = h2 - h1;             //ht2 - ht1
		x[(t << 2) + 3] = h1 - x[(t << 2) + 3];//ht1 - ht3
	}
}

#endif