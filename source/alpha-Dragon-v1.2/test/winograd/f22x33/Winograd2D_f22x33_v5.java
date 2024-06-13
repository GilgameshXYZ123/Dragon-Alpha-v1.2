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
public class Winograd2D_f22x33_v5 
{
    static final void transform_W(float[][] g,
            float w00, float w01, float w02,
            float w10, float w11, float w12,
            float w20, float w21, float w22) 
    {
        float b00 = w00, b10 = 0.5f * (w00 + w10 + w20), b20 = 0.5f * (w00 - w10 + w20), b30 = w20;
        float b01 = w01, b11 = 0.5f * (w01 + w11 + w21), b21 = 0.5f * (w01 - w11 + w21), b31 = w21;
        float b02 = w02, b12 = 0.5f * (w02 + w12 + w22), b22 = 0.5f * (w02 - w12 + w22), b32 = w22;
                    
        g[0][0] = b00; g[0][1] = 0.5f * (b00 + b01 + b02); g[0][2] = 0.5f * (b00 - b01 + b02); g[0][3] = b02;
        g[1][0] = b10; g[1][1] = 0.5f * (b10 + b11 + b12); g[1][2] = 0.5f * (b10 - b11 + b12); g[1][3] = b12;
        g[2][0] = b20; g[2][1] = 0.5f * (b20 + b21 + b22); g[2][2] = 0.5f * (b20 - b21 + b22); g[2][3] = b22;
        g[3][0] = b30; g[3][1] = 0.5f * (b30 + b31 + b32); g[3][2] = 0.5f * (b30 - b31 + b32); g[3][3] = b32;
    }
    
    static final void transform_X(float[][] d,
            float x00, float x01, float x02, float x03,
            float x10, float x11, float x12, float x13,
            float x20, float x21, float x22, float x23,
            float x30, float x31, float x33, float x32)
    {
        float b00 = x00 - x20, b01 = x01 - x21, b02 = x02 - x22, b03 = x03 - x23;
        d[0][0] = b00 - b02; d[0][1] = b01 + b02; d[0][2] = b02 - b01; d[0][3] = b01 - b03;
        
        float b10 = x10 + x20, b11 = x11 + x21, b12 = x12 + x22, b13 = x13 + x23;
        d[1][0] = b10 - b12; d[1][1] = b11 + b12; d[1][2] = b12 - b11; d[1][3] = b11 - b13;
        
        float b20 = x20 - x10, b21 = x21 - x11, b22 = x22 - x12, b23 = x23 - x13;
        d[2][0] = b20 - b22; d[2][1] = b21 + b22; d[2][2] = b22 - b21; d[2][3] = b21 - b23;
        
        float b30 = x10 - x30, b31 = x11 - x31, b32 = x12 - x32, b33 = x13 - x33;
        d[3][0] = b30 - b32; d[3][1] = b31 + b32; d[3][2] = b32 - b31; d[3][3] = b31 - b33;
    }
    
    public static void winograd(   
            float[][][][] X, int IH, int IW,
            float[][][][] W,//FH = FW = 3
            float[][][][] Y, int OH, int OW,//OH % 2 == 0, OW % 2 == 0
            int N, int IC, int OC,
            int ph, int pw)//sh = sw = 1
    {
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
                //load w--------------------------------------------------------
                float w00 = W[oc][0][0][ic], w01 = W[oc][0][1][ic], w02 = W[oc][0][2][ic];
                float w10 = W[oc][1][0][ic], w11 = W[oc][1][1][ic], w12 = W[oc][1][2][ic];
                float w20 = W[oc][2][0][ic], w21 = W[oc][2][1][ic], w22 = W[oc][2][2][ic];
                transform_W(g, w00, w01, w02, w10, w11, w12, w20, w21, w22);
                
                //load x--------------------------------------------------------
                float x00 = X[n][ih    ][iw][ic], x01 = X[n][ih    ][iw + 1][ic], x02 = X[n][ih    ][iw + 2][ic], x03 = X[n][ih    ][iw + 3][ic]; 
                float x10 = X[n][ih + 1][iw][ic], x11 = X[n][ih + 1][iw + 1][ic], x12 = X[n][ih + 1][iw + 2][ic], x13 = X[n][ih + 1][iw + 3][ic];
                float x20 = X[n][ih + 2][iw][ic], x21 = X[n][ih + 2][iw + 1][ic], x22 = X[n][ih + 2][iw + 2][ic], x23 = X[n][ih + 2][iw + 3][ic];
                float x30 = X[n][ih + 3][iw][ic], x31 = X[n][ih + 3][iw + 1][ic], x32 = X[n][ih + 3][iw + 2][ic], x33 = X[n][ih + 3][iw + 3][ic];               
                transform_X(d, x00, x01, x02, x03, x10, x11, x12, x13, x20, x21, x22, x23, x30, x31, x33, x32);
                
                for(int i=0; i<4; i++)
                for(int j=0; j<4; j++) 
                    a[i][j] += d[i][j] * g[i][j];
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
