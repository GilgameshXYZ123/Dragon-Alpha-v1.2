#pragma once

//Winograd F(2, 3)
#ifndef MICRO_WINOGRAD_F2X3_H
#define MICRO_WINOGRAD_F2X3_H

#define winograd_f2x3_add(v, g, d) {\
	float m1 = g.x * d.x;\
	float m2 = g.y * d.y;\
	float m3 = g.z * d.z;\
	float m4 = g.w * d.w;\
	v.x += (m1 + m2 + m3);\
	v.y += (m2 - m3 - m4); }\

#define winograd_f2x3_y(y, a) {\
	y[0] = a[0] + a[1] + a[2];\
	y[1] = a[1] - a[2] - a[3];}

#define winograd_f2x3_y_f32_64(o, a) {\
	o.x = a.x + a.y + a.z;\
	o.y = a.y - a.z - a.w;}


#define winograd_f2x3_g(g0, g1, g2)     float4{ g0, 0.5f*(g0 + g1 + g2), 0.5f*(g0 - g1 + g2), g2 }
#define winograd_f2x3_d(d0, d1, d2, d3) float4{ d0 - d2, d1 + d2, d2 - d1, d1 - d3 }


//G.y = 0.5f*(g0 + g1 + g2)
//G.z = 0.5f*(g0 - g1 + g2) = G.y - g1
#define WinoGrad_produce_G(G, g0, g1, g2) {\
	float v = 0.5f*(g0 + g1 + g2);\
	G = float4{ g0, v, v - g1, g2 }; }


#define winograd_f2x3_simdMM4(v0, v1, dv0, dv1, g0, g2) \
	{ (v0).x += (g0).x*(dv0); (v0).y += (g0).z*(dv0); (v0).z += (g2).x*(dv0); (v0).w += (g2).z*(dv0);\
	  (v1).x += (g0).y*(dv1); (v1).y += (g0).w*(dv1); (v1).z += (g2).y*(dv1); (v1).w += (g2).w*(dv1); }


//v0 += t0, v1 += t0
//v0 += t1, v2 -= t1
#define winograd_f2x3_VT4(v0, v1, t0, t1) {\
	v0.x += t0.x; v0.y += t0.y; v0.z += t0.z; v0.w += t0.w;\
	v1.x += t0.x; v1.y += t0.y; v1.z += t0.z; v1.w += t0.w;\
	v0.x += t1.x; v0.y += t1.y; v0.z += t1.z; v0.w += t1.w; \
	v1.x -= t1.x; v1.y -= t1.y; v1.z -= t1.z; v1.w -= t1.w;}

#endif