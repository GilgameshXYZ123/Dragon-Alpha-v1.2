#pragma once

#ifndef MICRO_DCONV3D_DX_IDX_H
#define MICRO_DCONV3D_DX_IDX_H

//GK(oc, fh, fw) -> GK(oc, index = [fh, fw])
//(fh, fw) -> Idx = (fh*4 + fw)

//for 3*3 filter: STEP = 4
//(0, 0) -> 0; (0, 1) -> 1; (0, 2) ->  2
//(1, 0) -> 4; (1, 1) -> 5; (1, 2) ->  6
//(2, 0) -> 8; (2, 1) -> 9; (2, 2) -> 10
__device__ __constant__ char YIDX_W33[] = {//STEP = 4
	0, 1, 2,
	4, 5, 6,
	8, 9, 10 };

//WIdx = fhr*3 + fwr
//[(0, 0), (0, 1), (0, 2)] -> [(2, 2), (2, 1), (2, 0)] -> [2*3+2, 2*3+1, 2*3 + 0] -> [8, 7, 6]
//[(1, 0), (1, 1), (1, 2)] -> [(1, 2), (1, 1), (0, 2)] -> [1*3+2, 1*3+1, 0*3 + 0] -> [5, 4, 3]
//[(2, 0), (2, 1), (2, 2)] -> [(0, 2), (0, 1), (0, 0)] -> [2, 1, 0]
__device__ __constant__ char WIDX_W33[] = {
	8, 7, 6,
	5, 4, 3,
	2, 1, 0 };

__device__ __constant__ char YIDX_V2_W3P1[]{//stride = 9, STEP = 4
	0, 1, 4, 5, 0, 0, 0, 0,  0,//tFH = 2, tFW = 3, fhw_idx = (0, 0)
	0, 1, 2, 4, 5, 6, 0, 0,  0,//tFH = 2, tFW = 3, fhw_idx = (0, 1)

	0, 1, 4, 5, 8, 9, 0, 0,  0,//tFH = 3, tFW = 2, fhw_idx = (1, 0)
	0, 1, 2, 4, 5, 6, 8, 9, 10 //tFH = 3, tFW = 3, fhw_idx = (1, 1)
};

__device__ __constant__ char YIDX_V2_W3P2[]{//stride = 9, STEP = 4
	0, 0, 0, 0, 0, 0, 0, 0,  0,//tFH = 1, tFW = 1, fhw_idx = (0, 0)
	0, 1, 0, 0, 0, 0, 0, 0,  0,//tFH = 1, tFW = 2, fhw_idx = (0, 1)
	0, 1, 2, 0, 0, 0, 0, 0,  0,//tFH = 1, tFW = 3, fhw_idx = (0, 2)

	0, 4, 0, 0, 0, 0, 0, 0,  0,//tFH = 2, tFW = 1, fhw_idx = (1, 0)
	0, 1, 4, 5, 0, 0, 0, 0,  0,//tFH = 2, tFW = 2, fhw_idx = (1, 1)
	0, 1, 2, 4, 5, 6, 0, 0,  0,//tFH = 2, tFW = 3, fhw_idx = (1, 2)

	0, 4, 9, 0, 0, 0, 0, 0,  0,//tFH = 3, tFW = 1, fhw_idx = (2, 0)
	0, 1, 4, 5, 8, 9, 0, 0,  0,//tFH = 3, tFW = 2, fhw_idx = (2, 1)
	0, 1, 2, 4, 5, 6, 8, 9, 10 //tFH = 3, tFW = 3, fhw_idx = (2, 2)
};

__device__ __constant__ char YIDX_V2_W3P2_FH[]{//stride = 9, STEP = 4
	0, 0, 0, 0, 0, 0, 0, 0, 0,//tFH = 1, tFW = 1, fhw_idx = (0, 0)
	0, 0, 0, 0, 0, 0, 0, 0, 0,//tFH = 1, tFW = 2, fhw_idx = (0, 1)
	0, 0, 0, 0, 0, 0, 0, 0, 0,//tFH = 1, tFW = 3, fhw_idx = (0, 2)

	0, 1, 0, 0, 0, 0, 0, 0, 0,//tFH = 2, tFW = 1, fhw_idx = (1, 0)
	0, 0, 1, 1, 0, 0, 0, 0, 0,//tFH = 2, tFW = 2, fhw_idx = (1, 1)
	0, 0, 0, 1, 1, 1, 0, 0, 0,//tFH = 2, tFW = 3, fhw_idx = (1, 2)

	0, 1, 2, 0, 0, 0, 0, 0, 0,//tFH = 3, tFW = 1, fhw_idx = (2, 0)
	0, 0, 1, 1, 2, 2, 0, 0, 0,//tFH = 3, tFW = 2, fhw_idx = (2, 1)
	0, 0, 0, 1, 1, 1, 2, 2, 2 //tFH = 3, tFW = 3, fhw_idx = (2, 2)
};

//for 5*5 Filter, STEP = 8
//(0, 0) ->  0; (0, 1) ->  1; (0, 2)->   2; (0, 3) ->  3; (0, 4) ->  4
//(1, 0) ->  8; (1, 1) ->  9; (1, 2) -> 10; (1, 3) -> 11; (1, 4) -> 12
//(2, 0) -> 16; (2, 1) -> 17; (2, 2) -> 18; (2, 3) -> 19; (2, 4) -> 20
//(3, 0) -> 24; (3, 1) -> 25; (3, 2) -> 26; (3, 3) -> 27; (3, 4) -> 28
//(4, 0) -> 32; (4, 1) -> 33; (4, 2) -> 34; (4, 3) -> 35; (4, 4) -> 36
__device__ __constant__ char YIDX_W55[] = {//STEP = 8
	 0,  1,  2,  3,  4,
	 8,  9, 10, 11, 12,
	16, 17, 18, 19, 20,
	24, 25, 26, 27, 28,
	32, 33, 34, 35, 36 };

//WIdx = fhr*5 + fwr
//[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)] -> [(4, 4), (4, 3), (4, 2), (4, 1), (4, 0)] -> [24, 23, 22, 21, 20]
//[(1, 0), (1, 1), (1, 2), (1, 3), (1, 4)] -> [(3, 4), (3, 3), (3, 2), (3, 1), (3, 0)] -> [19, 18, 17, 16, 15]
//[(2, 0), (2, 1), (2, 2), (2, 3), (2, 4)] -> [(2, 4), (2, 3), (2, 2), (2, 1), (2, 0)] -> [14, 13, 12, 11, 10]
//[(3, 0), (3, 1), (3, 2), (3, 3), (3, 4)] -> [(1, 4), (1, 3), (1, 2), (1, 1), (1, 0)] -> [ 9,  8,  7,  6,  5]
//[(4, 0), (4, 1), (4, 2), (4, 3), (4, 4)] -> [(0, 4), (0, 3), (0, 2), (0, 1), (0, 0)] -> [ 4,  3,  2,  1,  0]
__device__ __constant__ char WIDX_W55[] = {
	24, 23, 22, 21, 20,
	19, 18, 17, 16, 15,
	14, 13, 12, 11, 10,
	 9,  8,  7,  6,  5,
	 4,  3,  2,  1,  0 };

__device__ __constant__ char YIDX_V2_W5P2[] = {//stride = 25
	0, 1, 2, 8, 9, 10, 16, 17, 18,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,//tFH = 3, tFW = 3, fhw_idx = (0, 0)
	0, 1, 2, 3, 8,  9, 10, 11, 16, 17, 18, 19,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,//tFH = 3, tFW = 4, fhw_idx = (0, 1)
	0, 1, 2, 3, 4,  8,  9, 10, 11, 12, 16, 17, 18, 19, 20,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,//tFH = 3, tFW = 5, fhw_idx = (0, 2)

	0, 1, 2, 8, 9, 10, 16, 17, 18, 24, 25, 26,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,//tFH = 4, tFW = 3, fhw_idx = (1, 0)
	0, 1, 2, 3, 8,  9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27,  0,  0,  0,  0,  0,  0,  0,  0,  0,//tFH = 4, tFW = 4, fhw_idx = (1, 1)
	0, 1, 2, 3, 4,  8,  9, 10, 11, 12, 16, 17, 18, 19, 20, 24, 25, 26, 27, 28,  0,  0,  0,  0,  0,//tFH = 4, tFW = 5, fhw_idx = (1, 2)

	0, 1, 2, 8, 9, 10, 16, 17, 18, 24, 25, 26, 32, 33, 34,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,//tFH = 5, tFW = 3, fhw_idx = (2, 0)
	0, 1, 2, 3, 8,  9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27, 32, 33, 34, 35,  0,  0,  0,  0,  0,//tFH = 5, tFW = 4, fhw_idx = (2, 1)
	0, 1, 2, 3, 4,  8,  9, 10, 11, 12, 16, 17, 18, 19, 20, 24, 25, 26, 27, 28, 32, 33, 34, 35, 36,//tFH = 5, tFW = 5, fhw_idx = (2, 2)
};

#endif