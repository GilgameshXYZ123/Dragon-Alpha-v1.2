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
public class Winograd2D_f33x22_v1 
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
            
            float a00 = 0.0f, a01 = 0.0f, a02 = 0.0f, a03 = 0.0f;
            float a10 = 0.0f, a11 = 0.0f, a12 = 0.0f, a13 = 0.0f;
            float a20 = 0.0f, a21 = 0.0f, a22 = 0.0f, a23 = 0.0f;
            float a30 = 0.0f, a31 = 0.0f, a32 = 0.0f, a33 = 0.0f;
            
            for(int ic=0; ic<IC; ic++) 
            {
                //load and transform W------------------------------------------
                float[] w = new float[4]; {
                    w[0] = W[oc][0][0][ic]; w[1] = W[oc][0][1][ic];
                    w[2] = W[oc][1][0][ic]; w[3] = W[oc][1][1][ic];
                }
                
                float[] b = new float[8]; {//4*2
                    b[0] = w[0]; b[2] = 0.5f*(w[0] + w[2]); b[4] = 0.5f*(w[0] - w[2]); b[6] = w[2];
                    b[1] = w[1]; b[3] = 0.5f*(w[1] + w[3]); b[5] = 0.5f*(w[1] - w[3]); b[7] = w[3];
                }
                
                float g00, g01, g02, g03;
                float g10, g11, g12, g13;
                float g20, g21, g22, g23;
                float g30, g31, g32, g33; {
                    g00 = b[0]; g01 = 0.5f*(b[0] + b[1]); g02 = 0.5f*(b[0] - b[1]); g03 = b[1];
                    g10 = b[2]; g11 = 0.5f*(b[2] + b[3]); g12 = 0.5f*(b[2] - b[3]); g13 = b[3]; 
                    g20 = b[4]; g21 = 0.5f*(b[4] + b[5]); g22 = 0.5f*(b[4] - b[5]); g23 = b[5];
                    g30 = b[6]; g31 = 0.5f*(b[6] + b[7]); g32 = 0.5f*(b[6] - b[7]); g33 = b[7];
                }
                
                //load and transform X------------------------------------------
                float[] x = new float[16]; {//4*4
                    for(int i=0; i<4; i++)
                    for(int j=0; j<4; j++)
                        x[i*4 + j] = ld(X, n, ih + i, iw + j, ic, IH, IW); 
                }
                
                float h00, h01, h02, h03;
                float h10, h11, h12, h13;
                float h20, h21, h22, h23;
                float h30, h31, h32, h33; {
                    h00 = x[0] - x[ 8]; h10 = x[4] + x[ 8]; h20 = x[ 8] - x[4]; h30 = x[12] - x[4];
                    h01 = x[1] - x[ 9]; h11 = x[5] + x[ 9]; h21 = x[ 9] - x[5]; h31 = x[13] - x[5];
                    h02 = x[2] - x[10]; h12 = x[6] + x[10]; h22 = x[10] - x[6]; h32 = x[14] - x[6];
                    h03 = x[3] - x[11]; h13 = x[7] + x[11]; h23 = x[11] - x[7]; h33 = x[15] - x[7];
                }
                
                float d00, d01, d02, d03;
                float d10, d11, d12, d13;
                float d20, d21, d22, d23;
                float d30, d31, d32, d33; {
                    d00 = h00 - h02; d01 = h01 + h02; d02 = h02 - h01; d03 = h03 - h01;
                    d10 = h10 - h12; d11 = h11 + h12; d12 = h12 - h11; d13 = h13 - h11;
                    d20 = h20 - h22; d21 = h21 + h22; d22 = h22 - h21; d23 = h23 - h21;
                    d30 = h30 - h32; d31 = h31 + h32; d32 = h32 - h31; d33 = h33 - h31;
                }
                
                //accumulate----------------------------------------------------
                a00 += g00*d00; a01 += g01*d01; a02 += g02*d02; a03 += g03*d03;
                a10 += g10*d10; a11 += g11*d11; a12 += g12*d12; a13 += g13*d13;
                a20 += g20*d20; a21 += g21*d21; a22 += g22*d22; a23 += g23*d23;
                a30 += g30*d30; a31 += g31*d31; a32 += g32*d32; a33 += g33*d33;
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
