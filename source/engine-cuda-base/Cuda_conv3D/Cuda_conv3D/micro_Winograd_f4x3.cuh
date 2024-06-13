#pragma once

#ifndef MICRO_WINOGRAD_F4X3_H
#define MICRO_WINOGRAD_F4X3_H

#define winograd_f4x3_d(d0, d1, d2, d3, d4, d5, x0, x1, x2, x3, x4, x5) {\
	d0 = 24.0f * x0 - 30.0f * x2 + 6.0f * x4;\
    d1 =  16.0f * x1 + 16.0f * x2 - 4.0f * x3 - 4.0f * x4;\
    d2 = -16.0f * x1 + 16.0f * x2 + 4.0f * x3 - 4.0f * x4;\
    d3 = -2.0f * x1 - x2 + 2.0f * x3 + x4;\
	d4 =  2.0f * x1 - x2 - 2.0f * x3 + x4;\
    d5 = 96.0f * x1 - 120.0f * x3 + 24.0f * x5; }

#define winograd_f4x3_g(g0, g1, g2, g3, g4, g5, w0, w1, w2) {\
	g0 = w0;\
	g1 = w0 + w1 + w2;\
	g2 = w0 - w1 + w2;\
	g3 = w0 + 2 * w1 + 4 * w2;\
	g4 = w0 - 2 * w1 + 4 * w2;\
	g5 = w2;}


#define winograd_f4x3_v(v0, v1, v2, v3, m0, m1, m2, m3, m4, m5) {\
	v0 = 0.0416667f * (m0 + m1 + m2 + m3 + m4);\
	v1 = 0.0416667f * (m1 - m2 + 2.0f * m3 - 2.0f * m4);\
	v2 = 0.0416667f * (m1 + m2 + 4.0f * m3 + 4.0f * m4);\
	v3 = 0.0416667f * (m1 - m2 + 8.0f * m3 - 8.0f * m4 + m5); }

#define winograd_f4x3_v2(v0, v1, v2, v3, m1, m2, m3, m4, m5) {\
	v0 = 0.0416667f * (v0 + m1 + m2 + m3 + m4);\
	v1 = 0.0416667f * (m1 - m2 + 2.0f * m3 - 2.0f * m4);\
	v2 = 0.0416667f * (m1 + m2 + 4.0f * m3 + 4.0f * m4);\
	v3 = 0.0416667f * (m1 - m2 + 8.0f * m3 - 8.0f * m4 + m5); }

#define winograd_f4x3_v3(v0, v1, v2, v3, m1, m2, m3, m4) {\
	v0 = 0.0416667f * (v0 + m1 + m2 + m3 + m4);\
	v1 = 0.0416667f * (m1 - m2 + 2.0f * m3 - 2.0f * m4);\
	v2 = 0.0416667f * (m1 + m2 + 4.0f * m3 + 4.0f * m4);\
	v3 = 0.0416667f * (m1 - m2 + 8.0f * m3 - 8.0f * m4 + v3); }


#define winograd_f4x3_g2(g0, g1, g2, g3, g4, g5, w0, w1, w2) {\
	g0 = 0.0416667f * w0;\
	g1 = 0.0416667f * (w0 + w1 + w2);\
	g2 = 0.0416667f * (w0 - w1 + w2);\
	g3 = 0.0416667f * (w0 + 2 * w1 + 4 * w2);\
	g4 = 0.0416667f * (w0 - 2 * w1 + 4 * w2);\
	g5 = 0.0416667f * w2;}

#define winograd_f4x3_incr(v0, v1, v2, v3, m1, m2, m3, m4) {\
	v0 += (v0 + m1 + m2 + m3 + m4);\
	v1 += (m1 - m2 + 2.0f * m3 - 2.0f * m4);\
	v2 += (m1 + m2 + 4.0f * m3 + 4.0f * m4);\
	v3 += (m1 - m2 + 8.0f * m3 - 8.0f * m4 + v3); }

#endif
