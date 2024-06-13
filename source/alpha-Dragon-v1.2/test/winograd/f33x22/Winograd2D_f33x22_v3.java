/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package winograd.f33x22;

/**
 *
 * @author Gilgamesh
 */
public class Winograd2D_f33x22_v3 
{
     public static float ld(float[][][][] X, int n, int h, int w, int c, int H, int W) {
        boolean flag = (h>=0) && (h<H) && (w>=0) && (w<W);
        return flag? X[n][h][w][c] : 0.0f;
    }
    
    public static void wt(float[][][][] Y, float v, int n, int h, int w, int c, int H, int W) {
        boolean flag = (h>=0) && (h<H) && (w>=0) && (w<W);
        if(flag) Y[n][h][w][c] = v;
    }
    
    public static void winograd(
            float[][][][] X, int IH, int IW,
            float[][][][] W,
            float[][][][] Y, int OH, int OW,//OH % 2 == 0, OW % 2 == 0
            int N, int IC, int OC,
            int ph, int pw)//sh = sw = 1
    {
        for(int n=0; n<N; n++)
        for(int oc=0; oc<OC; oc++)
        for(int oh=0; oh<OH; oh += 3)
        for(int ow=0; ow<OW; ow += 3)
        {
            int ih = oh - ph;
            int iw = ow - pw;
            
            float[][] a = new float[4][4];
            
            float a00 = 0.0f, a01 = 0.0f, a02 = 0.0f, a03 = 0.0f;
            float a10 = 0.0f, a11 = 0.0f, a12 = 0.0f, a13 = 0.0f;
            float a20 = 0.0f, a21 = 0.0f, a22 = 0.0f, a23 = 0.0f;
            float a30 = 0.0f, a31 = 0.0f, a32 = 0.0f, a33 = 0.0f;
            
            for(int ic=0; ic<IC; ic++) 
            {
                //load and transform W------------------------------------------
                float[] w = new float[4];//2*2 
                w[0] = W[oc][0][0][ic]; w[1] = W[oc][0][1][ic];
                w[2] = W[oc][1][0][ic]; w[3] = W[oc][1][1][ic];
                
                float[] b = new float[8];//4*2
                b[0] = w[0]; b[2] = 0.5f*(w[0] + w[2]); b[4] = 0.5f*(w[0] - w[2]); b[6] = w[2];
                b[1] = w[1]; b[3] = 0.5f*(w[1] + w[3]); b[5] = 0.5f*(w[1] - w[3]); b[7] = w[3];
                
                float[] g = new float[16];//4*4
                g[ 0] = b[0]; g[ 1] = 0.5f*(b[0] + b[1]); g[ 2] = 0.5f*(b[0] - b[1]); g[ 3] = b[1];
                g[ 4] = b[2]; g[ 5] = 0.5f*(b[2] + b[3]); g[ 6] = 0.5f*(b[2] - b[3]); g[ 7] = b[3]; 
                g[ 8] = b[4]; g[ 9] = 0.5f*(b[4] + b[5]); g[10] = 0.5f*(b[4] - b[5]); g[11] = b[5];
                g[12] = b[6]; g[13] = 0.5f*(b[6] + b[7]); g[14] = 0.5f*(b[6] - b[7]); g[15] = b[7];

                //load and transform X------------------------------------------
                float[] x = new float[16];//4*4
                for(int i=0; i<4; i++)
                for(int j=0; j<4; j++)
                    x[i*4 + j] = ld(X, n, ih + i, iw + j, ic, IH, IW); 
                
                for(int t=0; t<4; t++) {//for each column
                    float x1 = x[4 + t];
                    float x2 = x[8 + t];
                    x[     t] = x[t] - x2; 
                    x[ 4 + t] = x1 + x2;
                    x[ 8 + t] = x2 - x1;
                    x[12 + t] = x[12 + t] - x1;
                }   
            
                float d00, d01, d02, d03;
                float d10, d11, d12, d13;
                float d20, d21, d22, d23;
                float d30, d31, d32, d33; {
                    d00 = x[ 0] - x[ 2]; d01 = x[ 1] + x[ 2]; d02 = x[ 2] - x[ 1]; d03 = x[ 3] - x[1];
                    d10 = x[ 4] - x[ 6]; d11 = x[ 5] + x[ 6]; d12 = x[ 6] - x[ 5]; d13 = x[ 7] - x[5];
                    d20 = x[ 8] - x[10]; d21 = x[ 9] + x[10]; d22 = x[10] - x[ 9]; d23 = x[11] - x[9];
                    d30 = x[12] - x[14]; d31 = x[13] + x[14]; d32 = x[14] - x[13]; d33 = x[15] - x[13];
                }
                
                //accumulate----------------------------------------------------
                a00 += g[ 0]*d00; a01 += g[ 1]*d01; a02 += g[ 2]*d02; a03 += g[ 3]*d03;
                a10 += g[ 4]*d10; a11 += g[ 5]*d11; a12 += g[ 6]*d12; a13 += g[ 7]*d13;
                a20 += g[ 8]*d20; a21 += g[ 9]*d21; a22 += g[10]*d22; a23 += g[11]*d23;
                a30 += g[12]*d30; a31 += g[13]*d31; a32 += g[14]*d32; a33 += g[15]*d33;
            }
            
            //output transformation=========================================
            float k00, k01, k02, k03;
            float k10, k11, k12, k13;
            float k20, k21, k22, k23; {
                k00 = a00 + a10 + a20; k10 = a10 - a20; k20 = a10 + a20 + a30;
                k01 = a01 + a11 + a21; k11 = a11 - a21; k21 = a11 + a21 + a31;
                k02 = a02 + a12 + a22; k12 = a12 - a22; k22 = a12 + a22 + a32;
                k03 = a03 + a13 + a23; k13 = a13 - a23; k23 = a13 + a23 + a33;
            }
            
            float y00, y01, y02;
            float y10, y11, y12;
            float y20, y21, y22; {
                y00 = k00 + k01 + k02; y01 = k01 - k02; y02 = k01 + k02 + k03;
                y10 = k10 + k11 + k12; y11 = k11 - k12; y12 = k11 + k12 + k13;
                y20 = k20 + k21 + k22; y21 = k21 - k22; y22 = k21 + k22 + k23;
            }
            
            wt(Y, y00, n, oh, ow    , oc, OH, OW);
            wt(Y, y01, n, oh, ow + 1, oc, OH, OW);
            wt(Y, y02, n, oh, ow + 2, oc, OH, OW);
              
            wt(Y, y10, n, oh + 1, ow    , oc, OH, OW);
            wt(Y, y11, n, oh + 1, ow + 1, oc, OH, OW);
            wt(Y, y12, n, oh + 1, ow + 2, oc, OH, OW);
            
            wt(Y, y20, n, oh + 2, ow    , oc, OH, OW);
            wt(Y, y21, n, oh + 2, ow + 1, oc, OH, OW);
            wt(Y, y22, n, oh + 2, ow + 2, oc, OH, OW);
        }
    }
    
}
