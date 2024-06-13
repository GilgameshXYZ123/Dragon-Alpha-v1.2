#pragma once

#ifndef CONV_3D_WINOGRAD_2D_F22X33R_H
#define CONV_3D_WINOGRAD_2D_F22X33R_H

//F(2*3, 3*3): FH = FW = 3, 2*2 = 4 elements
#include "conv3D_Winograd2D_f22x33R_util.cuh"
#include "conv3D_Winograd2D_f22x33R_kernel.cuh"

//F(3*3, 2*2): FH = FW = 2, 3*3 = 9 elements
#include "conv3D_Winograd2D_f33x22R_util.cuh"
#include "conv3D_Winograd2D_f33x22R_kernel.cuh"

#endif