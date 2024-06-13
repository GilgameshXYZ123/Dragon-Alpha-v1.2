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
public class Winograd_f2x3_v2 
{
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
        for(int gm = 0; gm < GM; gm += 2)
        {
            int n = gm / (OH * OW), gmr = gm % (OH * OW);
            int oh = gmr / OW, ow = gmr % OW;//[n, oh, ow, oc], [n, oh, ow+1, oc]
           
            float v0 = 0, v1 = 0;
            for(int gk = 0; gk < GK; gk += 3)
            {
                int fh = gk / FW, fw0 = gk % FW;
                int fw1 = fw0 + 1, fw2 = fw0 + 2;
                
                int ih = oh - ph + fh;
                int iw0 = ow - pw + fw0;
                int iw1 = iw0 + 1;
                int iw2 = iw0 + 2;
                int iw3 = iw0 + 3;
               
                for(int ic=0; ic < IC; ic++) 
                {
                    float d0 = 0, d1 = 0, d2 = 0, d3 = 0;
                    if((ih >= 0) && (ih < IH) && (iw0 >= 0) && (iw0 < IW)) d0 = X[n][ih][iw0][ic];
                    if((ih >= 0) && (ih < IH) && (iw1 >= 0) && (iw1 < IW)) d1 = X[n][ih][iw1][ic];
                    if((ih >= 0) && (ih < IH) && (iw2 >= 0) && (iw2 < IW)) d2 = X[n][ih][iw2][ic];
                    if((ih >= 0) && (ih < IH) && (iw3 >= 0) && (iw3 < IW)) d3 = X[n][ih][iw3][ic];
                    
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
