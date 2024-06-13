#pragma once

#ifndef MICRO_H
#define MICRO_H

#define F32_2_0 float2{0, 0}
#define F32_4_0 float4{0, 0, 0, 0}

#define in_range1D(w, W) ((w>=0)&&(w<W))
#define in_range2D(h, w, H, W) ((h>=0)&&(h<H) && (w>=0)&&(w<W))
#define in_range3D(h, w, c, H, W, C) ((h>=0)&&(h<H) && (w>=0)&&(w<W) && (c>=0)&&(c<C))

#define IS_POWER2(x) ( ((x)>0) && ((x)&((x)-1))==0 )

#define get2d(A, y, x, stride)   A[(y)*(stride) + (x)]
#define lget2d(A, y, x, lstride) A[((y)<<(lstride)) + (x)] //lstride = log2(stride)

#define get3d(A, z, y, x, Sy, Sx)    A[(((z)*(Sy)) + (y))*(Sx) + (x)]
#define lget3d(A, z, y, x, Sy, LSx)  A[((((z)*(Sy)) + y)<<(LSx)) + (x)]

#define get4d(A, w, z, y, x, Sz, Sy, Sx)    A[(((w)*(Sz) + (z))*(Sy) + (y))*(Sx) + (x)]
#define get4d_s(A, w, z, y, Sz, Sy, Sx)  A[(((w)*(Sz) + (z))*(Sy) + (y))*(Sx)]
#define lget4d(A, w, z, y, x, Sz, LSy, LSx) A[(((((w)*(Sz) + (z))<<(LSy)) + (y))<<(LSx)) + (x)]

#define within_width(v, index4, stride, width) {\
	v.x *= ((index4    ) % stride) < width;\
	v.y *= ((index4 + 1) % stride) < width;\
	v.z *= ((index4 + 2) % stride) < width;\
	v.w *= ((index4 + 3) % stride) < width;}

#endif