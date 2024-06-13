/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package winograd.f22x33;

/**
 *
 * @author Gilgamesh
 */
public class Winograd2D_f22x33_va 
{
    static final void transform_W(
            float[][][][] W,//FH = FW = 3
            float[][][][] G,//FH = FW = 4;
            int OC, int IC) 
    {
        for(int oc=0; oc<OC; oc++)
        for(int ic=0; ic<IC; ic++) {
            float w00 = W[oc][0][0][ic], w01 = W[oc][0][1][ic], w02 = W[oc][0][2][ic];
            float w10 = W[oc][1][0][ic], w11 = W[oc][1][1][ic], w12 = W[oc][1][2][ic];
            float w20 = W[oc][2][0][ic], w21 = W[oc][2][1][ic], w22 = W[oc][2][2][ic];
            
            float b00 = w00, b10 = 0.5f * (w00 + w10 + w20), b20 = 0.5f * (w00 - w10 + w20), b30 = w20;
            float b01 = w01, b11 = 0.5f * (w01 + w11 + w21), b21 = 0.5f * (w01 - w11 + w21), b31 = w21;
            float b02 = w02, b12 = 0.5f * (w02 + w12 + w22), b22 = 0.5f * (w02 - w12 + w22), b32 = w22;
                    
            float g00 = b00, g01 = 0.5f * (b00 + b01 + b02), g02 = 0.5f * (b00 - b01 + b02), g03 = b02;
            float g10 = b10, g11 = 0.5f * (b10 + b11 + b12), g12 = 0.5f * (b10 - b11 + b12), g13 = b12;
            float g20 = b20, g21 = 0.5f * (b20 + b21 + b22), g22 = 0.5f * (b20 - b21 + b22), g23 = b22;
            float g30 = b30, g31 = 0.5f * (b30 + b31 + b32), g32 = 0.5f * (b30 - b31 + b32), g33 = b32;
            
            G[oc][0][0][ic] = g00; G[oc][0][1][ic] = g01; G[oc][0][2][ic] = g02; G[oc][0][3][ic] = g03;
            G[oc][1][0][ic] = g10; G[oc][1][1][ic] = g11; G[oc][1][2][ic] = g12; G[oc][1][3][ic] = g13;
            G[oc][2][0][ic] = g20; G[oc][2][1][ic] = g21; G[oc][2][2][ic] = g22; G[oc][2][3][ic] = g23;
            G[oc][3][0][ic] = g30; G[oc][3][1][ic] = g31; G[oc][3][2][ic] = g32; G[oc][3][3][ic] = g33;
        }
    }
    
    public static final void __sync_threads() {}
    
    public static void winograd(   
            float[][][][] X, int IH, int IW,
            float[][][][] W,//FH = FW = 3
            float[][][][] Y, int OH, int OW,//OH % 2 == 0, OW % 2 == 0
            int N, int IC, int OC,
            int ph, int pw)//sh = sw = 1
    {
        float[][][][] G = new float[OC][4][4][IC];
        transform_W(W, G, OC, IC);
        
        for(int n=0; n<N; n++)
        for(int oc=0; oc<OC; oc++)
        for(int oh=0; oh<OH; oh += 2)
        for(int ow=0; ow<OW; ow += 2)
        {
            int ih = oh - ph, iw = ow - pw;
            float[][] a = new float[4][4];
            
            float[][] d = new float[4][4];
            float[][] g = new float[4][4]; 
            for(int ic=0; ic<IC; ic++) {
                for(int t=0; t<4; t++) {//load w(4*4)
                    g[t][0] = G[oc][t][0][ic]; 
                    g[t][1] = G[oc][t][1][ic];
                    g[t][2] = G[oc][t][2][ic];
                    g[t][3] = G[oc][t][3][ic];
                    
                    float x0 = X[n][ih    ][iw + t][ic]; 
                    float x1 = X[n][ih + 1][iw + t][ic]; 
                    float x2 = X[n][ih + 2][iw + t][ic]; 
                    float x3 = X[n][ih + 3][iw + t][ic];
                    
                    d[0][t] = x0 - x2; 
                    d[1][t] = x1 + x2;
                    d[2][t] = x2 - x1;
                    d[3][t] = x1 - x3;
                }
                __sync_threads();
                
                //input transform-----------------------------------------------
                for(int t=0; t<4; t++) {
                    float b0 = d[t][0], b1 = d[t][1], b2 = d[t][2], b3 = d[t][3];
                    d[t][0] = b0 - b2; 
                    d[t][1] = b1 + b2;
                    d[t][2] = b2 - b1;
                    d[t][3] = b1 - b3;
                }
                __sync_threads();
                
                //Hadamard Product----------------------------------------------
                for(int i=0; i<4; i++)
                for(int j=0; j<4; j++) 
                    a[i][j] += d[i][j] * g[i][j];
                __sync_threads();
            }
            
            float[][] y = new float[2][2]; {
                float b00 = a[0][0] + a[1][0] + a[2][0], b10 = a[1][0] - a[2][0] - a[3][0];
                float b01 = a[0][1] + a[1][1] + a[2][1], b11 = a[1][1] - a[2][1] - a[3][1];
                float b02 = a[0][2] + a[1][2] + a[2][2], b12 = a[1][2] - a[2][2] - a[3][2];
                float b03 = a[0][3] + a[1][3] + a[2][3], b13 = a[1][3] - a[2][3] - a[3][3];
        
                y[0][0] = b00 + b01 + b02; y[0][1] = b01 - b02 - b03;
                y[1][0] = b10 + b11 + b12; y[1][1] = b11 - b12 - b13;
            }
            
            for(int i=0; i<2; i++)
            for(int j=0; j<2; j++) 
                Y[n][oh + i][ow + j][oc] = y[i][j];
        }
    }
}
