#pragma once

#ifndef COMMON_MICRO_H
#define COMMON_MICRO_H

__device__ float HOLE[260];
__device__ __constant__ float _ZERO = 0;//default value of zero is 0


#define GRID_MAX 8192

#define MAX_STREAM_SIZE_ZeroPadding 10
#define MAX_STREAM_SIZE_KernelSplit 10
#define MAX_STREAM_SIZE_CrossAdd 7


#define FLOAT_ZERO4 make_float4(0, 0, 0, 0)
#define FLOAT_ZERO2 make_float2(0, 0)

#define F32_2_0 float2{ 0, 0 }
#define F32_4_0 float4{ 0, 0, 0, 0 }


#define IS_POWER2(x) ( ((x)>0) && ((x)&((x)-1))==0 )


#define get2d(A, y, x, stride) A[(y)*(stride) + (x)]
#define get3d(A, z, y, x, Sy, Sx) A[(((z)*(Sy)) + (y))*(Sx) + (x)]
#define get4d(A, w, z, y, x, Sz, Sy, Sx) A[(((w)*(Sz) + (z))*(Sy) + (y))*(Sx) + (x)]


#define simdMM4(c, av, b) {(c).x += (av) * (b).x; (c).y += (av) * (b).y; (c).z += (av) * (b).z; (c).w += (av) * (b).w;}
#define simdMM2(c, av, b) {(c).x += (av) * (b).x; (c).y += (av) * (b).y;} 
#define simdAdd4(c, av, b) {(c).x = (av) + (b).x; (c).y = (av) + (b).y; (c).z = (av) + (b).z; (c).w = (av) + (b).w;}
#define Mul4(c, av, b) {(c).x = (av) * (b).x; (c).y = (av) * (b).y; (c).z = (av) * (b).z; (c).w = (av) * (b).w;}

#define MAX_V_INT(k0, k1) (((k0) > (k1))*((k0) - (k1)) + (k1))


//if: flag == 1: -flag = -1 = 0xffffffff
//if: flag == 0: -flag =  0 
#define IF_int(flag, a, b) ( ((-flag) & (a - b)) + b)


//X = flag*v
#define zero_float(X, flag, v) \
	{float fv = v; int iv = -(flag) & *(int*)(&fv); X = *(float*)(&iv); }

#define zero_float4(X, flag) \
	{ if(!flag) X.x = X.y = X.z = X.w = 0;}


#define GET_OH(IH, FH, sh, ph) ( ((IH) + ((ph) << 1) - (FH))/(sh) + 1 )
#define GET_OW(IW, FW, sw, pw) ( ((IW) + ((pw) << 1) - (FW))/(sw) + 1 )

#define GET_IH(OH, FH, sh, ph) ( ((OH) - 1)*(sh) + (FH) - 2*(ph) )
#define GET_IW(OW, FW, sw, pw) ( ((OW) - 1)*(sw) + (FW) - 2*(pw) )


#define LOAD_Y(tih, tiw, fh, fw)\
	((tih >= -fh) && (tih < OH - fh) && (tiw >= -fw) && (tiw < OW - fw))


#define next_cudaStream(stream, streams, index, length) \
	cudaStream_t stream = (cudaStream_t)(intptr_t)streams[index]; { index = (index + 1) % length; }


#endif
