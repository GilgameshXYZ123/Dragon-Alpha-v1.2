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
public class Winograd2D_f22x33_v1 
{
    static final float[][] matMul(float[][] A, float[][] B) {
        int N = A.length, K = A[0].length, M = B[0].length;
        float[][] C = new float[N][M];
        for(int n=0; n<N; n++)
        for(int m=0; m<M; m++) {
            float c = 0.0f;
            for(int k=0; k<K; k++) c += A[n][k] * B[k][m];
            C[n][m] = c;
        }
        return C;
    }    
    
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
    
    static final float[][] transform_W(float[][] w) {
        float[][] G = WMat.G, GT = WMat.GT;
        return matMul(matMul(G, w), GT);//g = (G*W) * G^T
    }
    
    static final float[][] transform_X(float[][] x) {
        float[][] BT = WMat.BT, B = WMat.B;
        return matMul(matMul(BT, x), B);//d = B^T * X * B
    }
    
    static final float[][] transform_Y(float[][] a) {
        float[][] AT = WMat.AT, A = WMat.A;
        return matMul(matMul(AT, a), A);//y = A^T * a * A
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
