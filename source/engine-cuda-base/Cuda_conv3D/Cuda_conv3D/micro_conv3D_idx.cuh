#pragma once

#ifndef MICRO_CONV3D_IDX
#define MICRO_CONV3D_IDX

//[1] GK(fh, fw, ic) -> GK(index = [fh, fw], ic)
//[2] (fh, fw) -> Idx = (fh*STEP + fw)

//FH * FW = 3 * 3
#ifndef CONV3D_IDX_3X3
#define CONV3D_IDX_3X3

//for: 3 * 3 Filter, STEP = 4
//(0, 0) -> 0; (0, 1) -> 1; (0, 2) ->  2
//(1, 0) -> 4; (1, 1) -> 5; (1, 2) ->  6
//(2, 0) -> 8; (2, 1) -> 9; (2, 2) -> 10
__device__ __constant__ char XIDX_W3[] = {
	0, 1,  2,
	4, 5,  6,
	8, 9, 10 };

__device__ __constant__ char XIDX_V2_W3P1[] = {//stride = 9,
	0, 1, 4, 5, 0, 0, 0, 0,  0,//tFH = 2, tFW = 2, fhw_idx = (0, 0)
	0, 1, 2, 4, 5, 6, 0, 0,  0,//tFH = 2, tFW = 3, fhw_idx = (0, 1)
	0, 1, 4, 5, 8, 9, 0, 0,  0,//tFH = 3, tFW = 2, fhw_idx = (1, 0)
	0, 1, 2, 4, 5, 6, 8, 9, 10,//tFH = 3, tFW = 3, fhw_idx = (1, 1)
};

__device__ __constant__ char XIDX_V2_W3P1_FH[] = {//stride = 9, fhw_idx / 4
	0, 0, 1, 1, 0, 0, 0, 0, 0,//tFH = 2, tFW = 2, fhw_idx = (0, 0)
	0, 0, 0, 1, 1, 1, 0, 0, 0,//tFH = 2, tFW = 3, fhw_idx = (0, 1)
	0, 0, 1, 1, 2, 2, 0, 0, 0,//tFH = 3, tFW = 2, fhw_idx = (1, 0)
	0, 0, 0, 1, 1, 1, 2, 2, 2,//tFH = 3, tFW = 3, fhw_idx = (1, 1)
};

#endif


//FH * FW = 4 * 4
#ifndef CONV3D_IDX_4X4
#define CONV3D_IDX_4X4

//for: 4 * 4 Filter, STEP = 4
//(0, 0) ->  0; (0, 1) ->  1; (0, 2) ->  2; (0, 3) ->  3
//(1, 0) ->  4; (1, 1) ->  5; (1, 2) ->  6; (1, 3) ->  7
//(2, 0) ->  8; (2, 1) ->  9; (2, 2) -> 10; (2, 3) -> 11
//(3, 0) -> 12; (3, 1) -> 13; (3, 2) -> 14; (3, 3) -> 15
__device__ __constant__ char XIDX_V2_W4P1[] = {//stride = 16
	0, 1, 2, 4, 5, 6, 8, 9, 10,  0,  0,  0,  0,  0,  0,  0,//tFH = 3, tFW = 3, fhw_idx = (0, 0)
	0, 1, 2, 3, 4, 5, 6, 7,  8,  9, 10, 11,  0,  0,  0,  0,//tFH = 3, tFW = 4, fhw_idx = (0, 1)
	0, 1, 2, 4, 5, 6, 8, 8, 10, 12, 13, 14,  0,  0,  0,  0,//tFH = 4, tFW = 3, fhw_idx = (1, 0)
	0, 1, 2, 3, 4, 5, 6, 7,  8,  9, 10, 11, 12, 13, 14, 15 //tFH = 3, tFW = 3, fhw_idx = (1, 1)
};

__device__ __constant__ char XIDX_V2_W4P1_FH[] = {//stride = 16, fhw_idx / 4
	0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0,//tFH = 3, tFW = 3, fhw_idx = (0, 0)
	0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 0, 0, 0, 0,//tFH = 3, tFW = 4, fhw_idx = (0, 1)
	0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 0, 0, 0, 0,//tFH = 4, tFW = 3, fhw_idx = (1, 0)
	0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,//tFH = 3, tFW = 3, fhw_idx = (1, 1)
};

#endif


//FH * FW = 5 * 5
#ifndef CONV3D_IDX_5X5
#define CONV3D_IDX_5X5

//for 5*5 Filter, STEP = 8
//(0, 0) ->  0; (0, 1) ->  1; (0, 2)->   2; (0, 3) ->  3; (0, 4) ->  4
//(1, 0) ->  8; (1, 1) ->  9; (1, 2) -> 10; (1, 3) -> 11; (1, 4) -> 12
//(2, 0) -> 16; (2, 1) -> 17; (2, 2) -> 18; (2, 3) -> 19; (2, 4) -> 20
//(3, 0) -> 24; (3, 1) -> 25; (3, 2) -> 26; (3, 3) -> 27; (3, 4) -> 28
//(4, 0) -> 32; (4, 1) -> 33; (4, 2) -> 34; (4, 3) -> 35; (4, 4) -> 36
__device__ __constant__ char XIDX_W5[] = {
	 0,  1,  2,  3,  4,
	 8,  9, 10, 11, 12,
	16, 17, 18, 19, 20,
	24, 25, 26, 27, 28,
	32, 33, 34, 35, 36
};

__device__ __constant__ char XIDX_V2_W5P2[] = {//stride = 25
	0, 1, 2, 8, 9, 10, 16, 17, 18,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,//tFH = 3, tFW = 3, fhw_idx = (0, 0)
	0, 1, 2, 3, 8,  9, 10, 11, 16, 17, 18, 19,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,//tFH = 3, tFW = 4, fhw_idx = (0, 1)
	0, 1, 2, 3, 4,  8,  9, 10, 11, 12, 16, 17, 18, 19, 20,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,//tFH = 3, tFW = 5, fhw_idx = (0, 2)

	0, 1, 2, 8, 9, 10, 16, 17, 18, 24, 25, 26,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,//tFH = 4, tFW = 3, fhw_idx = (1, 0)
	0, 1, 2, 3, 8,  9, 10, 11, 16, 17, 18, 19, 24 ,25, 26, 27,  0,  0,  0,  0,  0,  0,  0,  0,  0,//tFH = 4, tFW = 4, fhw_idx = (1, 1)
	0, 1, 2, 3, 4,  8,  9, 10, 11, 12, 16, 17, 18, 18, 20, 24, 25, 26, 27, 28,  0,  0,  0,  0,  0,//tFH = 4, tFW = 5, fhw_idx = (1, 2)

	0, 1, 2, 8, 9, 10, 16, 17, 18, 24, 25, 26, 32, 33, 34,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,//tFH = 5, tFW = 3, fhw_idx = (2, 0)
	0, 1, 2, 3, 8,  9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27, 32, 33, 34, 35,  0,  0,  0,  0,  0,//tFH = 5, tFW = 4, fhw_idx = (2, 1)
	0, 1, 2, 3, 4,  8,  9, 10, 11, 12, 16, 17, 18, 19, 20, 24, 25, 26, 27, 28, 32, 33, 34, 35, 36,//tFH = 5, tFW = 5, fhw_idx = (2, 2)
};

__device__ __constant__ char XIDX_V2_W5P2_FH[] = {//stride = 25, div = 8
	0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,//tFH = 3, tFW = 3, fhw_idx = (0, 0)
	0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,//tFH = 3, tFW = 4, fhw_idx = (0, 1)
	0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,//tFH = 3, tFW = 5, fhw_idx = (0, 2)

	0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,//tFH = 4, tFW = 3, fhw_idx = (1, 0)
	0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0,//tFH = 4, tFW = 4, fhw_idx = (1, 1)
	0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0,//tFH = 4, tFW = 5, fhw_idx = (1, 2)

	0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,//tFH = 5, tFW = 3, fhw_idx = (2, 0)
	0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 0, 0, 0, 0, 0,//tFH = 5, tFW = 4, fhw_idx = (2, 1)
	0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4,//tFH = 5, tFW = 5, fhw_idx = (2, 2)
};

#endif


//FH * FW = 5 * 5
#ifndef CONV3D_IDX_6X6
#define CONV3D_IDX_6X6

//for 5*5 Filter, STEP = 8
//(0, 0) ->  0; (0, 1) ->  1; (0, 2)->   2; (0, 3) ->  3; (0, 4) ->  4, (0, 5) ->  5
//(1, 0) ->  8; (1, 1) ->  9; (1, 2) -> 10; (1, 3) -> 11; (1, 4) -> 12, (1, 5) -> 13
//(2, 0) -> 16; (2, 1) -> 17; (2, 2) -> 18; (2, 3) -> 19; (2, 4) -> 20, (2, 5) -> 21
//(3, 0) -> 24; (3, 1) -> 25; (3, 2) -> 26; (3, 3) -> 27; (3, 4) -> 28, (3, 5) -> 29
//(4, 0) -> 32; (4, 1) -> 33; (4, 2) -> 34; (4, 3) -> 35; (4, 4) -> 36, (4, 5) -> 37
//(5, 0) -> 40; (5, 1) -> 41; (5, 2) -> 42; (5, 3) -> 43; (5, 4) -> 44, (5, 5) -> 45

__device__ __constant__ char XIDX_W6[] = {
	 0,  1,  2,  3,  4,  5,
	 8,  9, 10, 11, 12, 13,
	16, 17, 18, 19, 20, 21,
	24, 25, 26, 27, 28, 29,
	32, 33, 34, 35, 36, 37,
	40, 41, 42, 43, 44, 45
};

#endif


//FH * FW = 7 * 7
#ifndef CONV3D_IDX_7X7
#define CONV3D_IDX_7X7

//for 7*7 Filter, STEP = 8
//(0, 0) ->  0; (0, 1) ->  1; (0, 2)->   2; (0, 3) ->  3; (0, 4) ->  4, (0, 5) ->  5, (0, 6) -> 6
//(1, 0) ->  8; (1, 1) ->  9; (1, 2) -> 10; (1, 3) -> 11; (1, 4) -> 12, (1, 5) -> 13, (1, 6) -> 14
//.......
__device__ __constant__ char XIDX_W7[] = {
	 0,  1,  2,  3,  4,  5,  6,//0
	 8,  9, 10, 11, 12, 13, 14,//1
	16, 17, 18, 19, 20, 21, 22,//2
	24, 25, 26, 27, 28, 29, 30,//3
	32, 33, 34, 35, 36, 37, 38,//4
	40, 41, 42, 43, 44, 45, 46,//5
	48, 49, 50, 51, 52, 53, 54,//6
};

#endif

#endif
