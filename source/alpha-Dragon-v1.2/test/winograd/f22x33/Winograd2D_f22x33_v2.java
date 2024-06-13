/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package winograd.f22x33;

import static winograd.f22x33.Winograd2D_f22x33_v1.matMul;

/**
 *
 * @author Gilgamesh
 */
public class Winograd2D_f22x33_v2 
{
     static final void load_W(float[][][][] W, int oc, int ic, float[][] w) {
        for(int i=0; i<3; i++)
        for(int j=0; j<3; j++) 
            w[i][j] = W[oc][i][j][ic];
    }
    
    static final void load_X(float[][][][] X, int n, int ih, int iw, int ic, float[][] x) {
        for(int i=0; i<4; i++)
        for(int j=0; j<4; j++)
            x[i][j] = X[n][ih + i][iw + j][ic];
    }
    
    static final float[][] transform_W(float[][] w) {//g = (G*W) * G^T
        float[][] b = new float[4][3];
        b[0][0] = w[0][0];
        b[0][1] = w[0][1];
        b[0][2] = w[0][2];
        
        b[1][0] = 0.5f * (w[0][0] + w[1][0] + w[2][0]);
        b[1][1] = 0.5f * (w[0][1] + w[1][1] + w[2][1]);
        b[1][2] = 0.5f * (w[0][2] + w[1][2] + w[2][2]);
        
        b[2][0] = 0.5f * (w[0][0] - w[1][0] + w[2][0]);
        b[2][1] = 0.5f * (w[0][1] - w[1][1] + w[2][1]);
        b[2][2] = 0.5f * (w[0][2] - w[1][2] + w[2][2]);
        
        b[3][0] = w[2][0];
        b[3][1] = w[2][1];
        b[3][2] = w[2][2];
        
        float[][] g = new float[4][4];
        g[0][0] = b[0][0];
        g[0][1] = 0.5f * (b[0][0] + b[0][1] + b[0][2]);
        g[0][2] = 0.5f * (b[0][0] - b[0][1] + b[0][2]);
        g[0][3] = b[0][2];
        
        g[1][0] = b[1][0];
        g[1][1] = 0.5f * (b[1][0] + b[1][1] + b[1][2]);
        g[1][2] = 0.5f * (b[1][0] - b[1][1] + b[1][2]);
        g[1][3] = b[1][2];
        
        g[2][0] = b[2][0];
        g[2][1] = 0.5f * (b[2][0] + b[2][1] + b[2][2]);
        g[2][2] = 0.5f * (b[2][0] - b[2][1] + b[2][2]);
        g[2][3] = b[2][2];
        
        g[3][0] = b[3][0];
        g[3][1] = 0.5f * (b[3][0] + b[3][1] + b[3][2]);
        g[3][2] = 0.5f * (b[3][0] - b[3][1] + b[3][2]);
        g[3][3] = b[3][2];
        return g;
    }
    
    static final float[][] transform_X(float[][] x) {//d = B^T * X * B
        float[][] b = new float[4][4];
        b[0][0] = x[0][0] - x[2][0];
        b[0][1] = x[0][1] - x[2][1];
        b[0][2] = x[0][2] - x[2][2];
        b[0][3] = x[0][3] - x[2][3];
        
        b[1][0] = x[1][0] + x[2][0];
        b[1][1] = x[1][1] + x[2][1];
        b[1][2] = x[1][2] + x[2][2];
        b[1][3] = x[1][3] + x[2][3];
        
        b[2][0] = x[2][0] - x[1][0];
        b[2][1] = x[2][1] - x[1][1];
        b[2][2] = x[2][2] - x[1][2];
        b[2][3] = x[2][3] - x[1][3];
        
        b[3][0] = x[1][0] - x[3][0];
        b[3][1] = x[1][1] - x[3][1];
        b[3][2] = x[1][2] - x[3][2];
        b[3][3] = x[1][3] - x[3][3];
        
        float[][] d = new float[4][4];
        d[0][0] = b[0][0] - b[0][2];
        d[0][1] = b[0][1] + b[0][2];
        d[0][2] = b[0][2] - b[0][1];
        d[0][3] = b[0][1] - b[0][3];
        
        d[1][0] = b[1][0] - b[1][2];
        d[1][1] = b[1][1] + b[1][2];
        d[1][2] = b[1][2] - b[1][1];
        d[1][3] = b[1][1] - b[1][3];
        
        d[2][0] = b[2][0] - b[2][2];
        d[2][1] = b[2][1] + b[2][2];
        d[2][2] = b[2][2] - b[2][1];
        d[2][3] = b[2][1] - b[2][3];
        
        d[3][0] = b[3][0] - b[3][2];
        d[3][1] = b[3][1] + b[3][2];
        d[3][2] = b[3][2] - b[3][1];
        d[3][3] = b[3][1] - b[3][3];
        return d;
    }
    
    static final float[][] transform_Y(float[][] a) {
        float[][] b = new float[2][4];
        b[0][0] = a[0][0] + a[1][0] + a[2][0];
        b[0][1] = a[0][1] + a[1][1] + a[2][1];
        b[0][2] = a[0][2] + a[1][2] + a[2][2];
        b[0][3] = a[0][3] + a[1][3] + a[2][3];
        
        b[1][0] = a[1][0] - a[2][0] - a[3][0];
        b[1][1] = a[1][1] - a[2][1] - a[3][1];
        b[1][2] = a[1][2] - a[2][2] - a[3][2];
        b[1][3] = a[1][3] - a[2][3] - a[3][3];
        
        float[][] y = new float[2][2];
        y[0][0] = b[0][0] + b[0][1] + b[0][2];
        y[0][1] = b[0][1] - b[0][2] - b[0][3];
        
        y[1][0] = b[1][0] + b[1][1] + b[1][2];
        y[1][1] = b[1][1] - b[1][2] - b[1][3];
        return y;
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
            float a[][] = new float[4][4];
            
            for(int ic=0; ic<IC; ic++) 
            {
                float[][] w = new float[3][3]; load_W(W, oc, ic, w);
                float[][] x = new float[4][4]; load_X(X, n, ih, iw, ic, x);
                
                float[][] g = transform_W(w);
                float[][] d = transform_X(x);
                
                for(int i=0; i<4; i++)
                for(int j=0; j<4; j++) a[i][j] += d[i][j] * g[i][j];
            }
             
            float[][] y = transform_Y(a);
            for(int i=0; i<2; i++)
            for(int j=0; j<2; j++) 
                Y[n][oh + i][ow + j][oc] = y[i][j];
        }
    }
}
