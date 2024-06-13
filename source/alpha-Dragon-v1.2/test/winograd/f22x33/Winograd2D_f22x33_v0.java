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
public class Winograd2D_f22x33_v0 
{
    public static final float load_X(int IH, int IW, int ih, int iw, float v) {
        if(ih>=0 && iw>=0 && ih<IH && iw<IW) return v;
        return 0;
    }
    
    public static void check_idx(float[][][][] X, int N, int IH, int IW, int IC) {
        for(int n=0; n<N; n++)
        for(int ih=0; ih<IH; ih++)
        for(int iw=0; iw<IW; iw++)
        for(int ic=0; ic<IC; ic++)
            if(X[n][ih][iw][ic] == 0) System.out.println(n + ", " + ih + ", " + iw + ", " + ic);
    }
    
    public static void winograd(
            float[][][][] X, int IH, int IW,
            float[][][][] W, int FH, int FW,//FH = FW = 3
            float[][][][] Y, int OH, int OW,//OH % 2 == 0, OW % 2 == 0
            int N, int IC, int OC,
            int ph, int pw)//sh = sw = 1
    {
        for(int n=0; n<N; n++)
        for(int oc=0; oc<OC; oc++)
        for(int oh=0; oh<OH; oh += 2)
        for(int ow=0; ow<OW; ow += 2)
        {
            float q00 = 0.0f, q01 = 0.0f, q02 = 0.0f, q03 = 0.0f;
            float q10 = 0.0f, q11 = 0.0f, q12 = 0.0f, q13 = 0.0f;
            float q20 = 0.0f, q21 = 0.0f, q22 = 0.0f, q23 = 0.0f;
            float q30 = 0.0f, q31 = 0.0f, q32 = 0.0f, q33 = 0.0f;
            
            int ih = oh - ph;
            int iw = ow - pw;
            for(int ic=0; ic<IC; ic++) 
            {
                float w00 = W[oc][0][0][ic], w01 = W[oc][0][1][ic], w02 = W[oc][0][2][ic];//W(oc, fh, fw, ic)
                float w10 = W[oc][1][0][ic], w11 = W[oc][1][1][ic], w12 = W[oc][1][2][ic];
                float w20 = W[oc][2][0][ic], w21 = W[oc][2][1][ic], w22 = W[oc][2][2][ic];
                
                float x00 = X[n][ih    ][iw][ic], x01 = X[n][ih    ][iw + 1][ic], x02 = X[n][ih    ][iw + 2][ic], x03 = X[n][ih    ][iw + 3][ic];
                float x10 = X[n][ih + 1][iw][ic], x11 = X[n][ih + 1][iw + 1][ic], x12 = X[n][ih + 1][iw + 2][ic], x13 = X[n][ih + 1][iw + 3][ic];
                float x20 = X[n][ih + 2][iw][ic], x21 = X[n][ih + 2][iw + 1][ic], x22 = X[n][ih + 2][iw + 2][ic], x23 = X[n][ih + 2][iw + 3][ic];
                float x30 = X[n][ih + 3][iw][ic], x31 = X[n][ih + 3][iw + 1][ic], x32 = X[n][ih + 3][iw + 2][ic], x33 = X[n][ih + 3][iw + 3][ic];
                
                //filter transformation=========================================
                float g00 = w00, g03 = w02;
                float g30 = w20, g33 = w22;
                
                float r0 = w00 + w02, g01 = 0.5f * (r0 + w01), g02 = 0.5f * (r0 - w01);
                float r1 = w00 + w20, g10 = 0.5f * (r1 + w10), g20 = 0.5f * (r1 - w10);
                float r2 = w01 + w21, v11 = 0.5f * (r2 + w11), v21 = 0.5f * (r2 - w11);
                float r3 = w02 + w22, g13 = 0.5f * (r3 + w12), g23 = 0.5f * (r3 - w12);
                float r4 = w20 + w22, g31 = 0.5f * (r4 + w21), g32 = 0.5f * (r4 - w21);
                
                float u0 = g10 + g13, g11 = 0.5f * (u0 + v11), g12 = 0.5f * (u0 - v11);
                float u1 = g20 + g23, g21 = 0.5f * (u1 + v21), g22 = 0.5f * (u1 - v21);
                
                //input transformation===========================================
                float h00 = x00 - x20, h01 = x01 - x21, h02 = x02 - x22, h03 = x03 - x23;
                float h10 = x10 + x20, h11 = x11 + x21, h12 = x12 + x22, h13 = x13 + x23;
                float h20 = x20 - x10, h21 = x21 - x11, h22 = x22 - x12, h23 = x23 - x13;
                float h30 = x10 - x30, h31 = x11 - x31, h32 = x12 - x32, h33 = x13 - x33;
               
                float b00 = h00 - h02, b01 = h01 + h02, b02 = h02 - h01, b03 = h01 - h03;
                float b10 = h10 - h12, b11 = h11 + h12, b12 = h12 - h11, b13 = h11 - h13;
                float b20 = h20 - h22, b21 = h21 + h22, b22 = h22 - h21, b23 = h21 - h23;
                float b30 = h30 - h32, b31 = h31 + h32, b32 = h31 - h32, b33 = h31 - h33;
                
                //acucmulate====================================================
                q00 += g00 * b00; q01 += g01 * b01; q02 += g02 * b02; q03 += g03 * b03;
                q10 += g10 * b10; q11 += g11 * b11; q12 += g12 * b12; q13 += g13 * b13;
                q20 += g20 * b20; q21 += g21 * b21; q22 += g22 * b22; q23 += g23 * b23;
                q30 += g30 * b30; q31 += g31 * b31; q32 += g32 * b32; q33 += g33 * b33;
            }
            
            //output transformation=========================================
            float k00 = q00 + q10 + q20, k01 = q01 + q11 + q21, k02 = q02 + q12 + q22, k03 = q03 + q13 + q23;
            float k10 = q10 - q20 - q30, k11 = q11 - q21 - q31, k12 = q12 - q22 - q32, k13 = q13 - q23 - q33;
            
            float y00 = k00 + k01 + k02, y01 = k01 - k02 - k03;
            float y10 = k10 + k11 + k12, y11 = k11 - k12 - k13;
            
//            print_idx(n, oh    , ow, oc, y00); print_idx(n, oh    , ow + 1, oc, y01);
//            print_idx(n, oh + 1, ow, oc, y10); print_idx(n, oh + 1, ow + 1, oc, y11);
            
            Y[n][oh    ][ow][oc] = y00; Y[n][oh    ][ow    ][oc] = y01;
            Y[n][oh + 1][ow][oc] = y10; Y[n][oh + 1][ow + 1][oc] = y11;
        }
    }
    
    public static void print_idx(int n, int oh, int ow, int oc, float v) {
        if(v == 0.0f) System.out.println(n + ", " + oh + ", " + ow + ", " + oc);
//        else System.out.println(v);
    }
}
