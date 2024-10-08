/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package winograd.f22x33;

import static winograd.f22x33.Winograd2D_f22x33_v1.load_W;
import static winograd.f22x33.Winograd2D_f22x33_v1.load_X;

/**
 *
 * @author Gilgamesh
 */
public class Winograd2D_f22x33_v3 
{
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
                //load w--------------------------------------------------------
                float[][] w = new float[3][3]; {
                    w[0][0] = W[oc][0][0][ic]; w[0][1] = W[oc][0][1][ic]; w[0][2] = W[oc][0][2][ic];
                    w[1][0] = W[oc][1][0][ic]; w[1][1] = W[oc][1][1][ic]; w[1][2] = W[oc][1][2][ic];
                    w[2][0] = W[oc][2][0][ic]; w[2][1] = W[oc][2][1][ic]; w[2][2] = W[oc][2][2][ic];
                }
                
                //load x--------------------------------------------------------
                float[][] x = new float[4][4]; {
                    x[0][0] = X[n][ih    ][iw][ic]; x[0][1] = X[n][ih    ][iw + 1][ic]; x[0][2] = X[n][ih    ][iw + 2][ic]; x[0][3] = X[n][ih    ][iw + 3][ic]; 
                    x[1][0] = X[n][ih + 1][iw][ic]; x[1][1] = X[n][ih + 1][iw + 1][ic]; x[1][2] = X[n][ih + 1][iw + 2][ic]; x[1][3] = X[n][ih + 1][iw + 3][ic];
                    x[2][0] = X[n][ih + 2][iw][ic]; x[2][1] = X[n][ih + 2][iw + 1][ic]; x[2][2] = X[n][ih + 2][iw + 2][ic]; x[2][3] = X[n][ih + 2][iw + 3][ic];
                    x[3][0] = X[n][ih + 3][iw][ic]; x[3][1] = X[n][ih + 3][iw + 1][ic]; x[3][2] = X[n][ih + 3][iw + 2][ic]; x[3][3] = X[n][ih + 3][iw + 3][ic];               
                }
                
                //transform: W -> g---------------------------------------------
                float[][] g = new float[4][4]; {
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
                }
                
                
                //transform: X -> d---------------------------------------------
                float[][] d = new float[4][4]; {
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
                }
                
                for(int i=0; i<4; i++)
                for(int j=0; j<4; j++) a[i][j] += d[i][j] * g[i][j];
            }
             
            float[][] yb = new float[2][4];
            yb[0][0] = a[0][0] + a[1][0] + a[2][0];
            yb[0][1] = a[0][1] + a[1][1] + a[2][1];
            yb[0][2] = a[0][2] + a[1][2] + a[2][2];
            yb[0][3] = a[0][3] + a[1][3] + a[2][3];
        
            yb[1][0] = a[1][0] - a[2][0] - a[3][0];
            yb[1][1] = a[1][1] - a[2][1] - a[3][1];
            yb[1][2] = a[1][2] - a[2][2] - a[3][2];
            yb[1][3] = a[1][3] - a[2][3] - a[3][3];
        
            float[][] y = new float[2][2];
            y[0][0] = yb[0][0] + yb[0][1] + yb[0][2];
            y[0][1] = yb[0][1] - yb[0][2] - yb[0][3];
        
            y[1][0] = yb[1][0] + yb[1][1] + yb[1][2];
            y[1][1] = yb[1][1] - yb[1][2] - yb[1][3];
            for(int i=0; i<2; i++)
            for(int j=0; j<2; j++) 
                Y[n][oh + i][ow + j][oc] = y[i][j];
        }
    }
    
}
