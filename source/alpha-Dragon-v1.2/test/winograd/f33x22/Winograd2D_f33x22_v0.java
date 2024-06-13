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
public class Winograd2D_f33x22_v0 
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
                float w00, w01;
                float w10, w11; {
                    w00 = W[oc][0][0][ic]; w01 = W[oc][0][1][ic];
                    w10 = W[oc][1][0][ic]; w11 = W[oc][1][1][ic];
                }
                
                float b00, b10, b20, b30;
                float b01, b11, b21, b31; {
                    b00 = w00; b10 = 0.5f*(w00 + w10); b20 = 0.5f*(w00 - w10); b30 = w10;
                    b01 = w01; b11 = 0.5f*(w01 + w11); b21 = 0.5f*(w01 - w11); b31 = w11;
                }
                
                float g00, g01, g02, g03;
                float g10, g11, g12, g13;
                float g20, g21, g22, g23;
                float g30, g31, g32, g33; {
                    g00 = b00; g01 = 0.5f*(b00 + b01); g02 = 0.5f*(b00 - b01); g03 = b01;
                    g10 = b10; g11 = 0.5f*(b10 + b11); g12 = 0.5f*(b10 - b11); g13 = b11; 
                    g20 = b20; g21 = 0.5f*(b20 + b21); g22 = 0.5f*(b20 - b21); g23 = b21;
                    g30 = b30; g31 = 0.5f*(b30 + b31); g32 = 0.5f*(b30 - b31); g33 = b31;
                }
                
                //load and transform X------------------------------------------
                float x00, x01, x02, x03;
                float x10, x11, x12, x13;
                float x20, x21, x22, x23; 
                float x30, x31, x32, x33; {
                    x00 = ld(X, n, ih, iw    , ic, IH, IW); 
                    x01 = ld(X, n, ih, iw + 1, ic, IH, IW); 
                    x02 = ld(X, n, ih, iw + 2, ic, IH, IW); 
                    x03 = ld(X, n, ih, iw + 3, ic, IH, IW); 
                    
                    x10 = ld(X, n, ih + 1, iw    , ic, IH, IW); 
                    x11 = ld(X, n, ih + 1, iw + 1, ic, IH, IW); 
                    x12 = ld(X, n, ih + 1, iw + 2, ic, IH, IW); 
                    x13 = ld(X, n, ih + 1, iw + 3, ic, IH, IW); 
                    
                    x20 = ld(X, n, ih + 2, iw    , ic, IH, IW); 
                    x21 = ld(X, n, ih + 2, iw + 1, ic, IH, IW); 
                    x22 = ld(X, n, ih + 2, iw + 2, ic, IH, IW); 
                    x23 = ld(X, n, ih + 2, iw + 3, ic, IH, IW); 
                    
                    x30 = ld(X, n, ih + 3, iw    , ic, IH, IW); 
                    x31 = ld(X, n, ih + 3, iw + 1, ic, IH, IW); 
                    x32 = ld(X, n, ih + 3, iw + 2, ic, IH, IW); 
                    x33 = ld(X, n, ih + 3, iw + 3, ic, IH, IW); 
                }
            
                float h00, h01, h02, h03;
                float h10, h11, h12, h13;
                float h20, h21, h22, h23;
                float h30, h31, h32, h33; {
                    h00 = x00 - x20; h10 = x10 + x20; h20 = x20 - x10; h30 = x30 - x10;
                    h01 = x01 - x21; h11 = x11 + x21; h21 = x21 - x11; h31 = x31 - x11;
                    h02 = x02 - x22; h12 = x12 + x22; h22 = x22 - x12; h32 = x32 - x12;
                    h03 = x03 - x23; h13 = x13 + x23; h23 = x23 - x13; h33 = x33 - x13;
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
