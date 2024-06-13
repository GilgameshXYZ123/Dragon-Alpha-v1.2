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
public class Winograd2D_f22x33_v8 
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
            float[][] d = new float[4][4];
            float[][] g = new float[4][4]; 
            float[][] a = new float[4][4];
            
            for(int ic=0; ic<IC; ic++) 
            {
                //load w(4*4)
                g[0][0] = G[oc][0][0][ic]; g[0][1] = G[oc][0][1][ic]; g[0][2] = G[oc][0][2][ic]; g[0][3] = G[oc][0][3][ic];
                g[1][0] = G[oc][1][0][ic]; g[1][1] = G[oc][1][1][ic]; g[1][2] = G[oc][1][2][ic]; g[1][3] = G[oc][1][3][ic];
                g[2][0] = G[oc][2][0][ic]; g[2][1] = G[oc][2][1][ic]; g[2][2] = G[oc][2][2][ic]; g[2][3] = G[oc][2][3][ic];
                g[3][0] = G[oc][3][0][ic]; g[3][1] = G[oc][3][1][ic]; g[3][2] = G[oc][3][2][ic]; g[3][3] = G[oc][3][3][ic];
                
                //load x(4*4)
                d[0][0] = X[n][ih][iw    ][ic]; d[1][0] = X[n][ih + 1][iw    ][ic]; d[2][0] = X[n][ih + 2][iw    ][ic]; d[3][0] = X[n][ih + 3][iw    ][ic];
                float b00 = d[0][0] - d[2][0], b10 = d[1][0] + d[2][0], b20 = d[2][0] - d[1][0], b30 = d[1][0] - d[3][0];
                
                d[0][1] = X[n][ih][iw + 1][ic]; d[1][1] = X[n][ih + 1][iw + 1][ic]; d[2][1] = X[n][ih + 2][iw + 1][ic]; d[3][1] = X[n][ih + 3][iw + 1][ic]; 
                float b01 = d[0][1] - d[2][1], b11 = d[1][1] + d[2][1], b21 = d[2][1] - d[1][1], b31 = d[1][1] - d[3][1];
                
                d[0][2] = X[n][ih][iw + 2][ic]; d[1][2] = X[n][ih + 1][iw + 2][ic]; d[2][2] = X[n][ih + 2][iw + 2][ic]; d[3][2] = X[n][ih + 3][iw + 2][ic];
                float b02 = d[0][2] - d[2][2], b12 = d[1][2] + d[2][2], b22 = d[2][2] - d[1][2], b32 = d[1][2] - d[3][2];
                
                d[0][3] = X[n][ih][iw + 3][ic]; d[1][3] = X[n][ih + 1][iw + 3][ic]; d[2][3] = X[n][ih + 2][iw + 3][ic]; d[3][3] = X[n][ih + 3][iw + 3][ic];
                float b03 = d[0][3] - d[2][3], b13 = d[1][3] + d[2][3], b23 = d[2][3] - d[1][3], b33 = d[1][3] - d[3][3];
                __sync_threads();
                
                //input transform-----------------------------------------------
                __sync_threads();
                
                d[0][0] = b00 - b02; d[0][1] = b01 + b02; d[0][2] = b02 - b01; d[0][3] = b01 - b03;
                d[1][0] = b10 - b12; d[1][1] = b11 + b12; d[1][2] = b12 - b11; d[1][3] = b11 - b13;
                d[2][0] = b20 - b22; d[2][1] = b21 + b22; d[2][2] = b22 - b21; d[2][3] = b21 - b23;
                d[3][0] = b30 - b32; d[3][1] = b31 + b32; d[3][2] = b32 - b31; d[3][3] = b31 - b33;
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
