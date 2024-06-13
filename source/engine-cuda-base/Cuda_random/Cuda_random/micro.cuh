#pragma once

#ifndef MICRO_H
#define MICRO_H

#ifndef COMMON_MICRO_H
#define COMMON_MICRO_H

#define GRID_MAX 8192
#define LENGTHV_MAX 4194303 //8192 * 512 - 1 > 8192 * 128

#define MAX_INT(x, y) ((x)^(((x)^(y))& -((x)<(y))))
#define MIN_INT(x, y) ((y)^(((x)^(y))& -((x)<(y))))

#define TWO_PI 6.28325f

#define FLOAT_MAX (3.4028235E38)
#define FLOAT_MIN (-3.4028234E38)

#define simdLinear4(b, alpha, a, beta) {\
	(b).x = (a).x *(alpha) + (beta);\
	(b).y = (a).y *(alpha) + (beta);\
	(b).z = (a).z *(alpha) + (beta);\
	(b).w = (a).w *(alpha) + (beta);}

#define simdMul4(c, a, b) {\
	(c).x = (a).x * (b).x;\
	(c).y = (a).y * (b).y;\
	(c).z = (a).z * (b).z;\
	(c).w = (a).w * (b).w; }

#define within_width(v, index4, stride, width) {\
	v.x *= ((index4    ) % stride) < width;\
	v.y *= ((index4 + 1) % stride) < width;\
	v.z *= ((index4 + 2) % stride) < width;\
	v.w *= ((index4 + 3) % stride) < width;}

#endif


#ifndef RANDOM_MICRO_H
#define RANDOM_MICRO_H

#define THREAD_MUL       32717u
#define THREAD_MUL_MOD  524287u

#define THREAD_ADD0   123u
#define THREAD_ADD1  2269u
#define THREAD_ADD2 12347u

#define THREAD_MOD0  4194303u // 4194303u = (1 << 22) - 1
#define THREAD_MOD1  8388607u // 8388607u = (1 << 23) - 1
#define THREAD_MOD2 16777215u //16777215u = (1 << 24) - 1

//generate a float belongs to (0, 1) depending on the seed
//(((seed) = (mul*(seed) + inc) & mod) / (mod + 1) )
//<1> original: (((seed) = ( 632229u*(seed) + 2100473u) &  4194303u) /  4194304.0f)
//<2> optim1:   (((seed) = ( 632229u*(seed) +   21473u) &  4194303u) /  4194304.0f)
//<3> optim2:   (((seed) = (  32083u*(seed) + 2100473u) &  4194303u) /  4194304.0f)
//<4> optim3:   (((seed) = (4148271u*(seed) + 2100473u) & 33554431u) / 33554432.0f)
//<5> optim4:   (((seed) = ( 632229u*(seed) + 2100473u) & 33554431u) / 33554432.0f)
#define NEXT_FLOAT(seed) \
	(((seed) = (251403917*(seed) + 2100473) & 1073741823) / 1073741823.0f)


//[v, p] belong to (0, 1): v = (v > p? : v1 : v2)
#define BERNOULI(v, p, v1, v2) \
	(((v)<=(p))*((v1)-(v2)) + (v2))

#define simdNextFloat4(v, seed) {\
	(v).x = NEXT_FLOAT(seed);\
	(v).y = NEXT_FLOAT(seed);\
	(v).z = NEXT_FLOAT(seed);\
	(v).w = NEXT_FLOAT(seed);}

#define simdNextFloat2(v, seed) {\
	(v).x = NEXT_FLOAT(seed);\
	(v).y = NEXT_FLOAT(seed);}

#endif

#endif