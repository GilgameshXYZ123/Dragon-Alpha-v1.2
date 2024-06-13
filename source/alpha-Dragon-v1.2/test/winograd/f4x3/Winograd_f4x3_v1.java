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
public class Winograd_f4x3_v1 
{
    public static void winograd(
            float[][][][] X, int IH, int IW,
            float[][][][] W, int FH, int FW,//FH = FW = 3
            float[][][][] Y, int OH, int OW,//OH % 2 == 0, OW % 2 == 0
            int N, int IC, int OC,
            int ph, int pw)//sh = sw = 1
    {
        int GM = N * OH * OW;
        int GK = FH * FW;
        for(int oc = 0; oc < OC; oc++)
        for(int gm = 0; gm < GM; gm += 4)
        {
            int n = gm / (OH * OW), gmr = gm % (OH * OW);
            int oh = gmr / OW, ow = gmr % OW;//[n, oh, ow, oc], [n, oh, ow+1, oc]
           
            float v0 = 0, v1 = 0, v2 = 0, v3 = 0;
            for(int gk = 0; gk < GK; gk += 3)
            for(int ic=0; ic < IC; ic++)
            {
                int fh = gk / FW, fw0 = gk % FW;
                int fw1 = fw0 + 1, fw2 = fw0 + 2, fw3 = fw0 + 3;
                
                int ih = oh - ph + fh;
                int iw0 = ow - pw + fw0;
                int iw1 = iw0 + 1, iw2 = iw0 + 2;
                int iw3 = iw0 + 3, iw4 = iw0 + 4, iw5 = iw0 + 5;
                
                boolean lx0 = (ih >= 0) && (ih < IH) && (iw0 >= 0) && (iw0 < IW);
                boolean lx1 = (ih >= 0) && (ih < IH) && (iw1 >= 0) && (iw1 < IW);
                boolean lx2 = (ih >= 0) && (ih < IH) && (iw2 >= 0) && (iw2 < IW);
                boolean lx3 = (ih >= 0) && (ih < IH) && (iw3 >= 0) && (iw3 < IW);
                boolean lx4 = (ih >= 0) && (ih < IH) && (iw4 >= 0) && (iw4 < IW);
                boolean lx5 = (ih >= 0) && (ih < IH) && (iw5 >= 0) && (iw5 < IW);
                
                //==============================================================
                float w0 = W[oc][fh][fw0][ic];
                float w1 = W[oc][fh][fw1][ic];
                float w2 = W[oc][fh][fw2][ic];
                
                float x0 = (lx0 ? X[n][ih][iw0][ic] : 0);
                float x1 = (lx1 ? X[n][ih][iw1][ic] : 0);
                float x2 = (lx2 ? X[n][ih][iw2][ic] : 0);
                float x3 = (lx3 ? X[n][ih][iw3][ic] : 0);
                float x4 = (lx4 ? X[n][ih][iw4][ic] : 0);
                float x5 = (lx5 ? X[n][ih][iw5][ic] : 0);
              
                //==============================================================
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
                
                float m0 = g0*d0;
                float m1 = g1*d1;
                float m2 = g2*d2;
                float m3 = g3*d3;
                float m4 = g4*d4;
                float m5 = g5*d5;
                
                v0 += m0 + m1 + m2 +   m3 +   m4;
                v1 +=      m1 - m2 + 2*m3 - 2*m4;
                v2 +=      m1 + m2 + 4*m3 + 4*m4;
                v3 +=      m1 - m2 + 8*m3 - 8*m4 + m5 ;
            }
            
           Y[n][oh][ow    ][oc] = v0 / 24;
           Y[n][oh][ow + 1][oc] = v1 / 24; 
           Y[n][oh][ow + 2][oc] = v2 / 24; 
           Y[n][oh][ow + 3][oc] = v3 / 24; 
        }
    }
}
