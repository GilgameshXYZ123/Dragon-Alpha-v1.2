/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package winograd.f4x3;

/**
 *
 * @author Gilgamesh
 */
public class Winograd_f4x3_v3 
{
    //[d0, d1, d2]   [g0]   [r0]
    //[d1, d2, d3] * [g1] = [r1]
    //[d2, d3, d4]   [g2]   [r2]
    //[d3. d4, d5]          [r3]
    public static void winograd(
            float[][][][] X, int IH, int IW,
            float[][][][] W, int FH, int FW,//FH = FW = 3
            float[][][][] Y, int OH, int OW,//OH % 2 == 0, OW % 2 == 0
            int N, int IC, int OC,
            int ph, int pw)//sh = sw = 1
    {
        int GM = N * OH * OW;
        for(int oc = 0; oc < OC; oc++)
        for(int gm = 0; gm < GM; gm += 4)
        {
            int n = gm / (OH * OW), gmr = gm % (OH * OW);
            int oh = gmr / OW, ow = gmr % OW;//[n, oh, ow, oc], [n, oh, ow+1, oc]
            
            int ih0 = oh - ph;
            int iw0 = ow - pw;
            
            float v0 = 0, v1 = 0, v2 = 0, v3 = 0;
            float m1 = 0, m2 = 0, m3 = 0, m4 = 0;
            
            for(int fh=0; fh<3; fh++)
            {
                boolean lh0 = (ih0 >= -fh) && (ih0 + fh < IH);
                boolean lx0 = lh0 && (iw0 >=  0) && (iw0     < IW);
                boolean lx1 = lh0 && (iw0 >= -1) && (iw0 + 1 < IW);
                boolean lx2 = lh0 && (iw0 >= -2) && (iw0 + 2 < IW);
                boolean lx3 = lh0 && (iw0 >= -3) && (iw0 + 3 < IW);
                boolean lx4 = lh0 && (iw0 >= -4) && (iw0 + 4 < IW);
                boolean lx5 = lh0 && (iw0 >= -5) && (iw0 + 5 < IW);
                
                for(int ic=0; ic<IC; ic++)
                {
                    float w0 = W[oc][fh][0][ic];
                    float w1 = W[oc][fh][1][ic];
                    float w2 = W[oc][fh][2][ic];
                    
                    float x0 = (lx0 ? X[n][ih0 + fh][iw0    ][ic] : 0);
                    float x1 = (lx1 ? X[n][ih0 + fh][iw0 + 1][ic] : 0);
                    float x2 = (lx2 ? X[n][ih0 + fh][iw0 + 2][ic] : 0);
                    float x3 = (lx3 ? X[n][ih0 + fh][iw0 + 3][ic] : 0);
                    float x4 = (lx4 ? X[n][ih0 + fh][iw0 + 4][ic] : 0);
                    float x5 = (lx5 ? X[n][ih0 + fh][iw0 + 5][ic] : 0);
                    
                    float g0 = w0;
                    float g1 = w0 + w1 + w2;
                    float g2 = w0 - w1 + w2;
                    float g3 = w0 + 2*w1 + 4*w2;
                    float g4 = w0 - 2*w1 + 4*w2;
                    float g5 = w2;
                    
                    float d0 = 24*x0 - 30*x2 + 6*x4;
                    float d1 =  16*x1 + 16*x2 - 4*x3 - 4*x4;
                    float d2 = -16*x1 + 16*x2 + 4*x3 - 4*x4;
                    float d3 = -2*x1 - x2 + 2*x3 + x4;
                    float d4 =  2*x1 - x2 - 2*x3 + x4;
                    float d5 = 96*x1 - 120*x3 + 24*x5;
                 
                    v0 += g0*d0; 
                    m1 += g1*d1; 
                    m2 += g2*d2;
                    m3 += g3*d3;
                    m4 += g4*d4;
                    v3 += g5*d5;
                }
            }
            
            v0 += m1 + m2 +   m3 +   m4;
            v1 += m1 - m2 + 2*m3 - 2*m4;
            v2 += m1 + m2 + 4*m3 + 4*m4;
            v3 += m1 - m2 + 8*m3 - 8*m4;
            
            Y[n][oh][ow    ][oc] = v0 / 24;
            Y[n][oh][ow + 1][oc] = v1 / 24; 
            Y[n][oh][ow + 2][oc] = v2 / 24; 
            Y[n][oh][ow + 3][oc] = v3 / 24; 
        }
    }
}
