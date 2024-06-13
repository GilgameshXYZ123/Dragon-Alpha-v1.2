/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package winograd.f2x3;

/**
 *
 * @author Gilgamesh
 */
public class Winograd_f2x3_v3 
{
    // AT*((G*g)(BT*d))
    
    //AT=
// ⎡1  1  1   1    1    1      1    0⎤
// ⎢                                 ⎥
// ⎢0  1  -1  2   -2   1/2   -1/2   0⎥
// ⎢                                 ⎥
// ⎢0  1  1   4    4   1/4    1/4   0⎥
// ⎢                                 ⎥
// ⎢0  1  -1  8   -8   1/8   -1/8   0⎥
// ⎢                                 ⎥
// ⎢0  1  1   16  16   1/16  1/16   0⎥
// ⎢                                 ⎥
// ⎣0  1  -1  32  -32  1/32  -1/32  1⎦
    
    //G = 
    //   1,    0,    0,
    //-2/9, -2/9, -2/9,
    //-2/9,  2/9, -2/9
    //1/90,  1/45, 2/45},
    //1/90, -1/45, 2/45},
    //1/45, 1/90, 1/180},
    //1/45, -/90, 1/180},
    //0.0f, 0.0f, 1.0f
    
    
    // BT = 
        // ⎡1   0    -21/4    0    21/4     0    -1  0⎤
        // ⎢                                          ⎥
        // ⎢0   1      1    -17/4  -17/4    1    1   0⎥
        // ⎢                                          ⎥
        // ⎢0   -1     1    17/4   -17/4   -1    1   0⎥
        // ⎢                                          ⎥
        // ⎢0  1/2    1/4   -5/2   -5/4     2    1   0⎥
        // ⎢                                          ⎥
        // ⎢0  -1/2   1/4    5/2   -5/4    -2    1   0⎥
        // ⎢                                          ⎥
        // ⎢0   2      4    -5/2    -5     1/2   1   0⎥
        // ⎢                                          ⎥
        // ⎢0   -2     4     5/2    -5    -1/2   1   0⎥
        // ⎢                                          ⎥
        // ⎣0   -1     0    21/4     0    -21/4  0   1⎦
    
    
    //[d0, d1, d2]   [g0]   [r0]
    //[d1, d2, d3] * [g1] = [r1]
    //               [g2]
    public static void winograd(
            float[][][][] X, int IH, int IW,
            float[][][][] W, int FH, int FW,//FH = FW = 3
            float[][][][] Y, int OH, int OW,//OH % 2 == 0, OW % 2 == 0
            int N, int IC, int OC,
            int ph, int pw)//sh = sw = 1
    {
        int GN = OC;
        int GM = N * OH * OW;
        int GK = FH * FW;
        
        for(int oc = 0; oc < GN; oc++)
        for(int gm = 0; gm < GM; gm += 6)
        {
            int n = gm / (OH * OW), gmr = gm % (OH * OW);
            int oh = gmr / OW, ow = gmr % OW;//[n, oh, ow, oc], [n, oh, ow+1, oc]
           
            float v0 = 0, v1 = 0, v2 = 0, v3 = 0, v4 = 0, v5 = 0;
            for(int fh=0; fh<3; fh++)
            for(int ic=0; ic<IC; ic++)
            {
                //W transform---------------------------------------------------
                float w0 = W[oc][fh][0][ic];
                float w1 = W[oc][fh][1][ic];
                float w2 = W[oc][fh][2][ic];
                
                float g0 = w0;                            //   1,     0     0,
                float g1 = (2.0f/9) * (-w0 - w1 - w2);    //-2/9,  -2/9, -2/9,
                float g2 = (2.0f/9) * (-w0 + w1 - w2);    //-2/9,   2/9, -2/9,
                float g3 = (1.0f/90) * (w0 + 2*w1 + 4*w2);//1/90,  1/45, 2/45,
                float g4 = (1.0f/90) * (w0 - 2*w1 + 4*w2);//1/90, -1/45, 2/45,
                float g5 = (1.0f/180) * (w0*4 - 2*w1 + w2);//1/45, 1/90, 1/180,
                float g6 = (1.0f/180) * (w0*4 - 2*w1 + w2);//1/45, -/90, 1/180,
                float g7 = w2;                             //   0,    0,     1,
                
                //X transform---------------------------------------------------
                float x0;
                float x1;
                float x2;
                float x3;
                float x4;
                float x5;
                float x6;
                float x7;
                
                float d0 = x0 + (-21.0f/4)*x2 + (21.0f/4)*x5 - x;// ⎡1   0    -21/4    0    21/4     0    -1  0⎤
                float d1;
                float d2;
                float d3;
                float d4;
                float d5;
                float d6;
                float d7;
                
                
                
                // ⎢                                          ⎥
                // ⎢0   1      1    -17/4  -17/4    1    1   0⎥
                // ⎢                                          ⎥
                // ⎢0   -1     1    17/4   -17/4   -1    1   0⎥
                // ⎢                                          ⎥
                // ⎢0  1/2    1/4   -5/2   -5/4     2    1   0⎥
                // ⎢                                          ⎥
                // ⎢0  -1/2   1/4    5/2   -5/4    -2    1   0⎥
                // ⎢                                          ⎥
                // ⎢0   2      4    -5/2    -5     1/2   1   0⎥
                // ⎢                                          ⎥
                // ⎢0   -2     4     5/2    -5    -1/2   1   0⎥
                // ⎢                                          ⎥
                // ⎣0   -1     0    21/4     0    -21/4  0   1⎦
            }
            
            
            for(int gk = 0; gk < GK; gk += 3)
            {
                int fh = gk / FW, fw0 = gk % FW;
                int fw1 = fw0 + 1, fw2 = fw0 + 2;
                
                int ih  = oh - ph + fh;
                int iw0 = ow - pw + fw0;
                int iw1 = iw0 + 1, iw2 = iw0 + 2, iw3 = iw0 + 3;
                
                boolean ld0 = (ih >= 0) && (ih < IH) && (iw0 >= 0) && (iw0 < IW);
                boolean ld1 = (ih >= 0) && (ih < IH) && (iw1 >= 0) && (iw1 < IW);
                boolean ld2 = (ih >= 0) && (ih < IH) && (iw3 >= 0) && (iw3 < IW);
                boolean ld3 = (ih >= 0) && (ih < IH) && (iw3 >= 0) && (iw3 < IW);
                
                for(int ic=0; ic < IC; ic++) 
                {
                    float d0 = (ld0 ? X[n][ih][iw0][ic] : 0);
                    float d1 = (ld1 ? X[n][ih][iw1][ic] : 0);
                    float d2 = (ld2 ? X[n][ih][iw2][ic] : 0);
                    float d3 = (ld3 ? X[n][ih][iw3][ic] : 0);
                    
                    float g0 = W[oc][fh][fw0][ic];
                    float g1 = W[oc][fh][fw1][ic];
                    float g2 = W[oc][fh][fw2][ic];
                    
                    float m1 = g0 * (d0 - d2);
                    float m2 = 0.5f * (d1 + d2) * (g0 + g1 + g2);
                    float m3 = 0.5f * (d2 - d1) * (g0 - g1 + g2);
                    float m4 = g2 * (d1 - d3);
                    
                    v0 += (m1 + m2 + m3);
                    v1 += (m2 - m3 - m4);
                }
            }
            
           Y[n][oh][ow][oc] = v0;
           Y[n][oh][ow + 1][oc] = v1; 
        }
    }
}
