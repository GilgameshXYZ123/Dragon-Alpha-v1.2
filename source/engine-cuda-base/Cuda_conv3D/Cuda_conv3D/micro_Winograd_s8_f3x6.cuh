#pragma once

//FW = 6
//group = 3 elements
#ifndef MICRO_WINOGRAD_F3X6_H
#define MICRO_WINOGRAD_F3X6_H

//======{ Filtrer-Transform: W(6) -> G(8) }==========
//[   1       0      0       0       0        0 ]
//[ -2/9    -2/9    -2/9   -2/9    -2/9    -2/9 ]
//[ -2/9     2/9    -2/9    2/9    -2/9     2/9 ]
//[  1/90    1/45   2/45    4/45    8/45   16/45]
//[  1/90   -1/45   2/45   -4/45    8/45  -16/45]
//[ 32/45   16/45   8/45    4/45    2/45    1/45]
//[ 32/45  -16/45   8/45   -4/45    2/45   -1/45]
//[   0       0       0       0      0       1 ]
#define winograd_f3x6_g(g, w0, w1, w2, w3, w4, w5) {\
	g[0] = w0;\
	g[1] = -0.22222222f * (w0 + w1 + w2 + w3 + w4 + w5);\
	g[2] = -0.22222222f * (w0 - w1 + w2 - w3 + w4 - w5);\
	g[3] = 0.011111111f*w0 + 0.022222222f*w1 + 0.044444444f*w2 + 0.088888889f*w3 + 0.17777778f *w4 + 0.35555556f* w5;\
	g[4] = 0.011111111f*w0 - 0.022222222f*w1 + 0.044444444f*w2 - 0.088888889f*w3 + 0.17777778f *w4 - 0.35555556f* w5;\
	g[5] = 0.71111111f *w0 + 0.35555556f *w1 + 0.17777778f *w2 + 0.088888889f*w3 + 0.044444444f*w4 + 0.022222222f*w5;\
	g[6] = 0.71111111f *w0 - 0.35555556f *w1 + 0.17777778f *w2 - 0.088888889f*w3 + 0.044444444f*w4 - 0.022222222f*w5;\
	g[7] = w5; }

//optimized: with less multiplication (22 -> 12)
#define winograd_f3x6_G(g, w0, w1, w2, w3, w4, w5) {\
	float t1, t2;\
	g[0] = w0;\
	t1 = -0.22222222f * (w0 + w2 + w4);\
	t2 = -0.22222222f * (w1 + w3 + w5);\
	g[1] = t1 + t2; g[2] = t1 - t2;\
	t1 = 0.011111111f*w0 + 0.044444444f*w2 + 0.17777778f*w4;\
	t2 = 0.022222222f*w1 + 0.088888889f*w3 + 0.35555556f*w5;\
	g[3] = t1 + t2; g[4] = t1 - t2;\
	t1 = 0.71111111f*w0 + 0.17777778f *w2 + 0.044444444f*w4;\
	t2 = 0.35555556f*w1 + 0.088888889f*w3 + 0.022222222f*w5;\
	g[5] = t1 + t2; g[6] = t1 - t2;\
	g[7] = w5; }


//======{ Input-Transform: X[8] -> D[8] }==========
//[1   0    -21/4    0    21/4     0    -1  0]
//[0    1     1    -17/4  -17/4    1    1   0]
//[0   -1     1     17/4  -17/4   -1    1   0]
//[0   1/2   1/4   -5/2   -5/4     2    1   0]
//[0  -1/2   1/4    5/2   -5/4    -2    1   0]
//[0    2     4    -5/2    -5     1/2   1   0]
//[0   -1     0    21/4     0    -21/4  0   1]
#define winograd_f3x6_d(d, x) {\
d[0] =  x[0]      - 5.25f*x[2]              + 5.25f*x[4]              - x[6];\
d[1] =       x[1]       + x[2] - 4.25f*x[3] - 4.25f*x[4]       + x[5] + x[6];\
d[2] =      -x[1]       + x[2] + 4.25f*x[3] - 4.25f*x[4]       - x[5] + x[6];\
d[3] =  0.5f*x[1] + 0.25f*x[2] - 2.5f *x[3] - 1.25f*x[4] + 2.0f *x[5] + x[6];\
d[4] = -0.5f*x[1] + 0.25f*x[2] + 2.5f *x[3] - 1.25f*x[4] - 2.0f *x[5] + x[6];\
d[5] =  2.0f*x[1] + 4.0f *x[2] - 2.5f *x[3] - 5.0f *x[4] + 0.5f *x[5] + x[6];\
d[6] = -2.0f*x[1] + 4.0f *x[2] + 2.5f *x[3] - 5.0f *x[4] - 0.5f *x[5] + x[6];\
d[7] = -     x[1]              + 5.25f*x[3]              - 5.25f*x[5] + x[7]; }

//optimized: with less multiplication (32 -> 16)
#define winograd_f3x6_D(d, x) {\
float t1, t2;\
d[0] = x[0] - 5.25f*(x[2] - x[4]) - x[6]; \
t1 = x[1] - 4.25f*x[3] + x[5]; \
t2 = x[2] - 4.25f*x[4] + x[6]; \
d[1] = t2 + t1; d[2] = t2 - t1;\
t1 = 0.5f *x[1] - 2.5f *x[3] + 2.0f *x[5];\
t2 = 0.25f*x[2] - 1.25f*x[4] +       x[6];\
d[3] = t2 + t1; d[4] = t2 - t1;\
t1 = 2.0f*x[1] - 2.5f *x[3] + 0.5f *x[5];\
t2 = 4.0f*x[2] - 5.0f *x[4] +       x[6];\
d[5] = t2 + t1; d[6] = t2 - t1;\
d[7] = - x[1] + 5.25f*(x[3] - x[5]) + x[7];}

#define winograd_f3x6_D_oft(d, x, oft) {\
float t1, t2;\
d[0] = x[0 + oft] - 5.25f*(x[2 + oft] - x[4 + oft]) - x[6 + oft];\
t1 = x[1 + oft] - 4.25f*x[3 + oft] + x[5 + oft]; \
t2 = x[2 + oft] - 4.25f*x[4 + oft] + x[6 + oft]; \
d[1] = t2 + t1; d[2] = t2 - t1;\
t1 = 0.5f *x[1 + oft] - 2.5f *x[3 + oft] + 2.0f *x[5 + oft];\
t2 = 0.25f*x[2 + oft] - 1.25f*x[4 + oft] +       x[6 + oft];\
d[3] = t2 + t1; d[4] = t2 - t1;\
t1 = 2.0f*x[1 + oft] - 2.5f *x[3 + oft] + 0.5f *x[5 + oft];\
t2 = 4.0f*x[2 + oft] - 5.0f *x[4 + oft] +       x[6 + oft];\
d[5] = t2 + t1; d[6] = t2 - t1;\
d[7] = - x[1 + oft] + 5.25f*(x[3 + oft] - x[5 + oft]) + x[7 + oft]; }


//======{ Ouput-Transform: A[8] -> Y[3] }==========
//[1  1   1   1   1    1     1    0]
//[0  1  -1   2  -2   1/2  -1/2   0]
//[0  1   1   4   4   1/4   1/4   1]
#define winograd_f3x6_y(y, a) {\
	y[0] = a[0] + a[1] + a[2] +      a[3] +      a[4] +       a[5] +       a[6];\
	y[1] =        a[1] - a[2] + 2.0f*a[3] - 2.0f*a[4] + 0.5f *a[5] - 0.5f *a[6];\
	y[2] =        a[1] + a[2] + 4.0f*a[3] + 4.0f*a[4] + 0.25f*a[5] + 0.25f*a[6] + a[7];}\

//optimized: with less multiplications
#define winograd_f3x6_Y(y, a) {\
	float t0 = a[1] + a[2], t1 = a[1] - a[2];\
	float t2 = a[3] + a[4], t3 = a[3] - a[4];\
	float t4 = a[5] + a[6], t5 = a[5] - a[6];\
	y[0] = a[0] + t0 +      t2 +       t4;\
	y[1] =        t1 + 2.0f*t3 + 0.5f *t5;\
	y[2] =        t0 + 4.0f*t2 + 0.25f*t4 + a[7];}

//optimized: with less multiplications
#define winograd_f3x6_Y_vec(y, a, elem) {\
	float t0 = a[1].elem + a[2].elem, t1 = a[1].elem - a[2].elem;\
	float t2 = a[3].elem + a[4].elem, t3 = a[3].elem - a[4].elem;\
	float t4 = a[5].elem + a[6].elem, t5 = a[5].elem - a[6].elem;\
	y[0] = a[0].elem + t0 +      t2 +       t4;\
	y[1] =             t1 + 2.0f*t3 + 0.5f *t5;\
	y[2] =             t0 + 4.0f*t2 + 0.25f*t4 + a[7].elem;}

#endif