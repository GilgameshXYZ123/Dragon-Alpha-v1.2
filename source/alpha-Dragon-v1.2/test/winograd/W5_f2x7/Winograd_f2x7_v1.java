/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package winograd.W5_f2x7;

import winograd.W5_f4x5.*;
import winograd.f6x3.*;
import winograd.f2x3.*;

/**
 *
 * @author Gilgamesh
 */
public class Winograd_f2x7_v1 
{
    // AT*((G*g)(BT*d))
    
//   AT = 
//       [1  1  1   1  1    1    1    0]
//       [0  1  -1  2  -2  1/2  -1/2  1]
        
//    G = 
//    [  1       0     0      0     0      0       0   ]
//    [-2/9    -2/9   -2/9  -2/9   -2/9   -2/9    -2/9 ]
//    [-2/9     2/9   -2/9   2/9   -2/9    2/9    -2/9 ]
//    [ 1/90    1/45   2/45   4/45  8/45   16/45  32/45]
//    [ 1/90   -1/45   2/45  -4/45  8/45  -16/45  32/45]
//    [32/45   16/45   8/45   4/45  2/45   1/45   1/90 ]
//    [32/45  -16/45   8/45  -4/45  2/45  -1/45   1/90 ]
//    [   0      0     0      0     0      0       1   ]

//    BT = 
//    ⎡1   0   -21/4    0      21/4     0    -1   s0⎤
//    ⎢0    1     1   -17/4   -17/4     1     1   0⎥
//    ⎢0   -1     1    17/4   -17/4    -1     1   0⎥
//    ⎢0   1/2   1/4   -5/2    -5/4     2     1   0⎥
//    ⎢0  -1/2   1/4    5/2    -5/4    -2     1   0⎥
//    ⎢0    2     4    -5/2     -5     1/2    1   0⎥
//    ⎢0   -2     4     5/2     -5    -1/2    1   0⎥
//    ⎣0   -1     0    21/4      0    -21/4   0   1⎦
   
    public static float ld(float[][][][] X, int n, int ih, int iw, int ic, int IH, int IW) {
        boolean flag = (ih>=0) && (ih<IH) && (iw>=0) && (iw<IW);
        if(flag) return X[n][ih][iw][ic];
        return 0;
    }
    
    public static void wt(float[][][][] Y, float v, int n, int oh, int ow, int oc, int OW) {
        boolean flag = ow < OW;
        if(flag) Y[n][oh][ow][oc] = v;
    }
    
    public static void winograd(
            float[][][][] X, int IH, int IW,
            float[][][][] W, int FH, int FW,//FH = FW = 3
            float[][][][] Y, int OH, int OW,//OH % 2 == 0, OW % 2 == 0
            int N, int IC, int OC,
            int ph, int pw)//sh = sw = 1
    {
        //(OH, OW) % 4 == 0
        for(int oc=0; oc<OC; oc++)
        for(int n=0; n<N; n++)
        for(int oh=0; oh<OH; oh++)
        for(int ow=0; ow<OW; ow+= 2)
        {
            float a0 = 0, a1 = 0, a2 = 0, a3 = 0;
            float a4 = 0, a5 = 0, a6 = 0, a7 = 0;
            
            for(int fh=0; fh<7; fh++)
            for(int ic=0; ic<IC; ic++)
            {
                //W transform---------------------------------------------------
                float w0 = W[oc][fh][0][ic];
                float w1 = W[oc][fh][1][ic];
                float w2 = W[oc][fh][2][ic];
                float w3 = W[oc][fh][3][ic];
                float w4 = W[oc][fh][4][ic];
                float w5 = W[oc][fh][5][ic];
                float w6 = W[oc][fh][6][ic];
                
                float g0 = w0;//[1      0     0      0      0   0,  0]
                float g1 = -0.22222222f * (w0 + w1 + w2 + w3 + w4 + w5 + w6);// [-2/9    -2/9   -2/9  -2/9   -2/9   -2/9    -2/9 ]
                float g2 = -0.22222222f * (w0 - w1 + w2 - w3 + w4 - w5 + w6);// [-2/9     2/9   -2/9   2/9   -2/9    2/9    -2/9 ]
                float g3 = 0.011111111f*w0 + 0.022222222f*w1 + 0.044444444f*w2 + 0.08888889f*w3 + 0.17777778f *w4 + 0.35555556f *w5 + 0.71111111f *w6;//[1/90    1/45   2/45   4/45  8/45   16/45  32/45]
                float g4 = 0.011111111f*w0 - 0.022222222f*w1 + 0.044444444f*w2 - 0.08888889f*w3 + 0.17777778f *w4 - 0.35555556f *w5 + 0.71111111f *w6;//[1/90   -1/45   2/45  -4/45  8/45  -16/45  32/45]
                float g5 = 0.7111111f  *w0 + 0.35555556f *w1 + 0.17777778f *w2 + 0.08888889f*w3 + 0.044444444f*w4 + 0.022222222f*w5 + 0.011111111f*w6;//[32/45   16/45   8/45   4/45  2/45   1/45   1/90 ]
                float g6 = 0.7111111f  *w0 - 0.35555556f *w1 + 0.17777778f *w2 - 0.08888889f*w3 + 0.044444444f*w4 - 0.022222222f*w5 + 0.011111111f*w6;//[32/45  -16/45   8/45  -4/45  2/45  -1/45   1/90 ]
                float g7 = w6;//[   0      0     0      0     0      0       1   ]
              
                //X transform---------------------------------------------------
                int ih = oh - ph + fh;
                int iw = ow - pw;
                        
                float x0 = ld(X, n, ih, iw + 0, ic, IH, IW);
                float x1 = ld(X, n, ih, iw + 1, ic, IH, IW);
                float x2 = ld(X, n, ih, iw + 2, ic, IH, IW);
                float x3 = ld(X, n, ih, iw + 3, ic, IH, IW);
                float x4 = ld(X, n, ih, iw + 4, ic, IH, IW);   
                float x5 = ld(X, n, ih, iw + 5, ic, IH, IW);
                float x6 = ld(X, n, ih, iw + 6, ic, IH, IW);
                float x7 = ld(X, n, ih, iw + 7, ic, IH, IW);
                
                //⎡1   0    -21/4    0    21/4     0    -1  0⎤
                float d0 = x0 - 5.25f*x2 + 5.25f*x4 - x6;
                
                //⎢0    1     1    -17/4  -17/4    1    1   0⎥
                //⎢0   -1     1     17/4  -17/4   -1    1   0⎥
                float d1 =  x1 + x2 - 4.25f*x3 - 4.25f*x4 + x5 + x6;
                float d2 = -x1 + x2 + 4.25f*x3 - 4.25f*x4 - x5 + x6;
                
                //⎢0   1/2   1/4   -5/2   -5/4     2    1   0⎥
                //⎢0  -1/2   1/4    5/2   -5/4    -2    1   0⎥
                float d3 =  0.5f*x1 + 0.25f*x2 - 2.5f*x3 - 1.25f*x4 + 2*x5 + x6;
                float d4 = -0.5f*x1 + 0.25f*x2 + 2.5f*x3 - 1.25f*x4 - 2*x5 + x6;
                
                //⎢0    2     4    -5/2    -5     1/2   1   0⎥
                //⎢0   -2     4     5/2    -5    -1/2   1   0⎥
                float d5 =  2*x1 + 4*x2 - 2.5f*x3 - 5*x4 + 0.5f*x5 + x6;
                float d6 = -2*x1 + 4*x2 + 2.5f*x3 - 5*x4 - 0.5f*x5 + x6;
                
                //⎣0   -1     0    21/4     0    -21/4  0   1⎦
                float d7 = -x1 + 5.25f*x3 - 5.25f*x5 + x7;
                
                a0 += g0*d0; 
                a1 += g1*d1;
                a2 += g2*d2; 
                a3 += g3*d3;
                a4 += g4*d4; 
                a5 += g5*d5; 
                a6 += g6*d6; 
                a7 += g7*d7;             
            }
            
            float v0 = a0 + a1 + a2 +    a3 +   a4 +      a5 +      a6;     //[1  1  1   1  1    1    1    0]
            float v1 =      a1 - a2 +  2*a3 - 2*a4 + 0.5f*a5 - 0.5f*a6 + a7;//[0  1  -1  2  -2  1/2  -1/2  1]
            
            wt(Y, v0, n, oh, ow    , oc, OW);
            wt(Y, v1, n, oh, ow + 1, oc, OW);
        }
    }
}
