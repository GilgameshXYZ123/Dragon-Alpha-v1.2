#pragma once

#ifndef MICRO_WINOGRAD_F3X2_H
#define MICRO_WINOGRAD_F3X2_H

#define winograd_f3x2_y(y, a) {\
	y[0] = a[0] + a[1] + a[2];\
	y[1] =        a[1] - a[2];\
	y[2] =        a[1] + a[2] + a[3];}

#define winograd_f3x2_y_f48_64(o, a) {\
	o.x = a.x + a.y + a.z;\
	o.y =       a.y - a.z;\
	o.z =       a.y + a.z + a.w; }

#endif