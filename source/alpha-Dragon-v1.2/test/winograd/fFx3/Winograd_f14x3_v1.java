/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package winograd.fFx3;

/**
 *
 * @author Gilgamesh
 */
public class Winograd_f14x3_v1 
{
    static float[][] AT = new float[][] {//14 * 16
        { 1.000000e+00f, 1.000000e+00f,  1.000000e+00f, 1.000000e+00f,  1.000000e+00f, 1.000000e+00f,  1.000000e+00f, 1.000000e+00f,  1.000000e+00f, 1.000000e+00f,  1.000000e+00f, 1.000000e+00f,  1.000000e+00f, 1.000000e+00f,  1.000000e+00f, 0.000000e+00f,  },
        { 0.000000e+00f, 1.000000e+00f, -1.000000e+00f, 2.000000e+00f, -2.000000e+00f, 5.000000e-01f, -5.000000e-01f, 3.000000e+00f, -3.000000e+00f, 3.333333e-01f, -3.333333e-01f, 4.000000e+00f, -4.000000e+00f, 2.500000e-01f, -2.500000e-01f, 0.000000e+00f,  },
        { 0.000000e+00f, 1.000000e+00f,  1.000000e+00f, 4.000000e+00f,  4.000000e+00f, 2.500000e-01f,  2.500000e-01f, 9.000000e+00f,  9.000000e+00f, 1.111111e-01f,  1.111111e-01f, 1.600000e+01f,  1.600000e+01f, 6.250000e-02f,  6.250000e-02f, 0.000000e+00f,  },
        { 0.000000e+00f, 1.000000e+00f, -1.000000e+00f, 8.000000e+00f, -8.000000e+00f, 1.250000e-01f, -1.250000e-01f, 2.700000e+01f, -2.700000e+01f, 3.703704e-02f, -3.703704e-02f, 6.400000e+01f, -6.400000e+01f, 1.562500e-02f, -1.562500e-02f, 0.000000e+00f,  },
        { 0.000000e+00f, 1.000000e+00f,  1.000000e+00f, 1.600000e+01f,  1.600000e+01f, 6.250000e-02f,  6.250000e-02f, 8.100000e+01f,  8.100000e+01f, 1.234568e-02f,  1.234568e-02f, 2.560000e+02f,  2.560000e+02f, 3.906250e-03f,  3.906250e-03f, 0.000000e+00f,  },
        { 0.000000e+00f, 1.000000e+00f, -1.000000e+00f, 3.200000e+01f, -3.200000e+01f, 3.125000e-02f, -3.125000e-02f, 2.430000e+02f, -2.430000e+02f, 4.115226e-03f, -4.115226e-03f, 1.024000e+03f, -1.024000e+03f, 9.765625e-04f, -9.765625e-04f, 0.000000e+00f,  },
        { 0.000000e+00f, 1.000000e+00f,  1.000000e+00f, 6.400000e+01f,  6.400000e+01f, 1.562500e-02f,  1.562500e-02f, 7.290000e+02f,  7.290000e+02f, 1.371742e-03f,  1.371742e-03f, 4.096000e+03f,  4.096000e+03f, 2.441406e-04f,  2.441406e-04f, 0.000000e+00f,  },
        { 0.000000e+00f, 1.000000e+00f, -1.000000e+00f, 1.280000e+02f, -1.280000e+02f, 7.812500e-03f, -7.812500e-03f, 2.187000e+03f, -2.187000e+03f, 4.572474e-04f, -4.572474e-04f, 1.638400e+04f, -1.638400e+04f, 6.103516e-05f, -6.103516e-05f, 0.000000e+00f,  },
        { 0.000000e+00f, 1.000000e+00f,  1.000000e+00f, 2.560000e+02f,  2.560000e+02f, 3.906250e-03f,  3.906250e-03f, 6.561000e+03f,  6.561000e+03f, 1.524158e-04f,  1.524158e-04f, 6.553600e+04f,  6.553600e+04f, 1.525879e-05f,  1.525879e-05f, 0.000000e+00f,  },
        { 0.000000e+00f, 1.000000e+00f, -1.000000e+00f, 5.120000e+02f, -5.120000e+02f, 1.953125e-03f, -1.953125e-03f, 1.968300e+04f, -1.968300e+04f, 5.080526e-05f, -5.080526e-05f, 2.621440e+05f, -2.621440e+05f, 3.814697e-06f, -3.814697e-06f, 0.000000e+00f,  },
        { 0.000000e+00f, 1.000000e+00f,  1.000000e+00f, 1.024000e+03f,  1.024000e+03f, 9.765625e-04f,  9.765625e-04f, 5.904900e+04f,  5.904900e+04f, 1.693509e-05f,  1.693509e-05f, 1.048576e+06f,  1.048576e+06f, 9.536743e-07f,  9.536743e-07f, 0.000000e+00f,  },
        { 0.000000e+00f, 1.000000e+00f, -1.000000e+00f, 2.048000e+03f, -2.048000e+03f, 4.882812e-04f, -4.882812e-04f, 1.771470e+05f, -1.771470e+05f, 5.645029e-06f, -5.645029e-06f, 4.194304e+06f, -4.194304e+06f, 2.384186e-07f, -2.384186e-07f, 0.000000e+00f,  },
        { 0.000000e+00f, 1.000000e+00f,  1.000000e+00f, 4.096000e+03f,  4.096000e+03f, 2.441406e-04f,  2.441406e-04f, 5.314410e+05f,  5.314410e+05f, 1.881676e-06f,  1.881676e-06f, 1.677722e+07f,  1.677722e+07f, 5.960464e-08f,  5.960464e-08f, 0.000000e+00f,  },
        { 0.000000e+00f, 1.000000e+00f, -1.000000e+00f, 8.192000e+03f, -8.192000e+03f, 1.220703e-04f, -1.220703e-04f, 1.594323e+06f, -1.594323e+06f, 6.272255e-07f, -6.272255e-07f, 6.710886e+07f, -6.710886e+07f, 1.490116e-08f, -1.490116e-08f, 1.000000e+00f,  },
    };
    
    static float[][] G = new float[][] {//16 * 3
        { 1.000000e+00f,   0.000000e+00f,  0.000000e+00f,  },
        { -2.222222e-03f, -2.222222e-03f, -2.222222e-03f,  },
        { -2.222222e-03f,  2.222222e-03f, -2.222222e-03f,  },
        {  1.209373e-05f,  2.418745e-05f,  4.837491e-05f,  },
        {  1.209373e-05f, -2.418745e-05f,  4.837491e-05f,  },
        {  1.981436e-01f,  9.907181e-02f,  4.953590e-02f,  },
        {  1.981436e-01f, -9.907181e-02f,  4.953590e-02f,  },
        { -2.854289e-07f, -8.562866e-07f, -2.568860e-06f,  },
        { -2.854289e-07f,  8.562866e-07f, -2.568860e-06f,  },
        { -1.365197e+00f, -4.550658e-01f, -1.516886e-01f,  },
        { -1.365197e+00f,  4.550658e-01f, -1.516886e-01f,  },
        {  6.218494e-09f,  2.487397e-08f,  9.949590e-08f,  },
        {  6.218494e-09f, -2.487397e-08f,  9.949590e-08f,  },
        {  1.669264e+00f,  4.173160e-01f,  1.043290e-01f,  },
        {  1.669264e+00f, -4.173160e-01f,  1.043290e-01f,  },
        {  0.000000e+00f,  0.000000e+00f,  1.000000e+00f,  },
    };
    
    static float[][] BT = new float[][] {//16 * 16
        { 1.000000e+00f,  0.000000e+00f, -3.042361e+01f,  0.000000e+00f,  2.857587e+02f,  0.000000e+00f, -9.371580e+02f,  0.000000e+00f,  9.371580e+02f,  0.000000e+00f, -2.857587e+02f,  0.000000e+00f,  3.042361e+01f,  0.000000e+00f, -1.000000e+00f, 0.000000e+00f,  },
        { 0.000000e+00f,  1.000000e+00f,  1.000000e+00f, -2.942361e+01f, -2.942361e+01f,  2.563351e+02f,  2.563351e+02f, -6.808229e+02f, -6.808229e+02f,  2.563351e+02f,  2.563351e+02f, -2.942361e+01f, -2.942361e+01f,  1.000000e+00f,  1.000000e+00f, 0.000000e+00f,  },
        { 0.000000e+00f, -1.000000e+00f,  1.000000e+00f,  2.942361e+01f, -2.942361e+01f, -2.563351e+02f,  2.563351e+02f,  6.808229e+02f, -6.808229e+02f, -2.563351e+02f,  2.563351e+02f,  2.942361e+01f, -2.942361e+01f, -1.000000e+00f,  1.000000e+00f, 0.000000e+00f,  },
        { 0.000000e+00f,  5.000000e-01f,  2.500000e-01f, -1.508681e+01f, -7.543403e+00f,  1.391076e+02f,  6.955382e+01f, -4.338021e+02f, -2.169010e+02f,  3.601285e+02f,  1.800642e+02f, -5.284722e+01f, -2.642361e+01f,  2.000000e+00f,  1.000000e+00f, 0.000000e+00f,  },
        { 0.000000e+00f, -5.000000e-01f,  2.500000e-01f,  1.508681e+01f, -7.543403e+00f, -1.391076e+02f,  6.955382e+01f,  4.338021e+02f, -2.169010e+02f, -3.601285e+02f,  1.800642e+02f,  5.284722e+01f, -2.642361e+01f, -2.000000e+00f,  1.000000e+00f, 0.000000e+00f,  },
        { 0.000000e+00f,  2.000000e+00f,  4.000000e+00f, -5.284722e+01f, -1.056944e+02f,  3.601285e+02f,  7.202569e+02f, -4.338021e+02f, -8.676042e+02f,  1.391076e+02f,  2.782153e+02f, -1.508681e+01f, -3.017361e+01f,  5.000000e-01f,  1.000000e+00f, 0.000000e+00f,  },
        { 0.000000e+00f, -2.000000e+00f,  4.000000e+00f,  5.284722e+01f, -1.056944e+02f, -3.601285e+02f,  7.202569e+02f,  4.338021e+02f, -8.676042e+02f, -1.391076e+02f,  2.782153e+02f,  1.508681e+01f, -3.017361e+01f, -5.000000e-01f,  1.000000e+00f, 0.000000e+00f,  },
        { 0.000000e+00f,  3.333333e-01f,  1.111111e-01f, -1.010417e+01f, -3.368056e+00f,  9.413021e+01f,  3.137674e+01f, -3.019271e+02f, -1.006424e+02f,  2.788385e+02f,  9.294618e+01f, -6.427083e+01f, -2.142361e+01f,  3.000000e+00f,  1.000000e+00f, 0.000000e+00f,  },
        { 0.000000e+00f, -3.333333e-01f,  1.111111e-01f,  1.010417e+01f, -3.368056e+00f, -9.413021e+01f,  3.137674e+01f,  3.019271e+02f, -1.006424e+02f, -2.788385e+02f,  9.294618e+01f,  6.427083e+01f, -2.142361e+01f, -3.000000e+00f,  1.000000e+00f, 0.000000e+00f,  },
        { 0.000000e+00f,  3.000000e+00f,  9.000000e+00f, -6.427083e+01f, -1.928125e+02f,  2.788385e+02f,  8.365156e+02f, -3.019271e+02f, -9.057812e+02f,  9.413021e+01f,  2.823906e+02f, -1.010417e+01f, -3.031250e+01f,  3.333333e-01f,  1.000000e+00f, 0.000000e+00f,  },
        { 0.000000e+00f, -3.000000e+00f,  9.000000e+00f,  6.427083e+01f, -1.928125e+02f, -2.788385e+02f,  8.365156e+02f,  3.019271e+02f, -9.057812e+02f, -9.413021e+01f,  2.823906e+02f,  1.010417e+01f, -3.031250e+01f, -3.333333e-01f,  1.000000e+00f, 0.000000e+00f,  },
        { 0.000000e+00f,  2.500000e-01f,  6.250000e-02f, -7.590278e+00f, -1.897569e+00f,  7.096528e+01f,  1.774132e+01f, -2.298542e+02f, -5.746354e+01f,  2.199236e+02f,  5.498090e+01f, -5.769444e+01f, -1.442361e+01f,  4.000000e+00f,  1.000000e+00f, 0.000000e+00f,  },
        { 0.000000e+00f, -2.500000e-01f,  6.250000e-02f,  7.590278e+00f, -1.897569e+00f, -7.096528e+01f,  1.774132e+01f,  2.298542e+02f, -5.746354e+01f, -2.199236e+02f,  5.498090e+01f,  5.769444e+01f, -1.442361e+01f, -4.000000e+00f,  1.000000e+00f, 0.000000e+00f,  },
        { 0.000000e+00f,  4.000000e+00f,  1.600000e+01f, -5.769444e+01f, -2.307778e+02f,  2.199236e+02f,  8.796944e+02f, -2.298542e+02f, -9.194167e+02f,  7.096528e+01f,  2.838611e+02f, -7.590278e+00f, -3.036111e+01f,  2.500000e-01f,  1.000000e+00f, 0.000000e+00f,  },
        { 0.000000e+00f, -4.000000e+00f,  1.600000e+01f,  5.769444e+01f, -2.307778e+02f, -2.199236e+02f,  8.796944e+02f,  2.298542e+02f, -9.194167e+02f, -7.096528e+01f,  2.838611e+02f,  7.590278e+00f, -3.036111e+01f, -2.500000e-01f,  1.000000e+00f, 0.000000e+00f,  },
        { 0.000000e+00f, -1.000000e+00f,  0.000000e+00f,  3.042361e+01f,  0.000000e+00f, -2.857587e+02f,  0.000000e+00f,  9.371580e+02f,  0.000000e+00f, -9.371580e+02f,  0.000000e+00f,  2.857587e+02f,  0.000000e+00f, -3.042361e+01f,  0.000000e+00f, 1.000000e+00f,  },
  
    };
    
    public static float ld(float[][][][] X, int n, int ih, int iw, int ic, int IH, int IW) {
        boolean flag = (ih>=0) && (ih<IH) && (iw>=0) && (iw<IW);
        if(flag) return X[n][ih][iw][ic];
        return 0;
    }
    
    public static void wt(float[][][][] Y, float v, int n, int oh, int ow, int oc, int OW) {
        boolean flag = ow < OW;
        if(flag) Y[n][oh][ow][oc] = v;
    }
    
    public static void matMul(float[][] C, float[][] A, float[][] B) {
        int N = C.length;
        int M = C[0].length;
        int K = A[0].length;
        
        for(int n=0; n<N; n++) 
        for(int m=0; m<M; m++) {
            float v = 0.0f;
            for(int k=0; k<K; k++) 
                v += A[n][k] * B[k][m];
            C[n][m] = v;
        }
    }
    
    public static void winograd(
            float[][][][] X, int IH, int IW,
            float[][][][] W, int FH, int FW,//FH = FW = 3
            float[][][][] Y, int OH, int OW,//OH % 2 == 0, OW % 2 == 0
            int N, int IC, int OC,
            int ph, int pw)//sh = sw = 1
    {
        for(int oc=0; oc<OC; oc++)
        for(int n=0; n<N; n++)
        for(int oh=0; oh<OH; oh++)
        for(int ow=0; ow<OW; ow+= 14)
        {
            float[][] a = new float[16][1]; //AT*((G*g)(BT*d)) = AT * accu
            for(int fh=0; fh<3; fh++)
            for(int ic=0; ic<IC; ic++)
            {
                //W transform---------------------------------------------------
                float w0 = W[oc][fh][0][ic];
                float w1 = W[oc][fh][1][ic];
                float w2 = W[oc][fh][2][ic];
                float[][] g = new float[16][1]; {
                    float  g0 = w0;  //1, 0,  0,
                    float  g1 = -2.222222e-03f * (w0 + w1 + w2);//-1/450, -1/450, -1/450
                    float  g2 = -2.222222e-03f * (w0 - w1 + w2);//-1/450,  1/450, -1/450
                    
                    float  g3 = 1.2093727E-5f  * (w0 + 2*w1 + 4*w2);//2/165375,  4/165375,  8/165375
                    float  g4 = 1.2093727E-5f * (w0 - 2*w1 + 4*w2);//2/165375, -4/165375,  8/165375
                    
                    float  g5 = 0.049535904f * (4*w0 + 2*w1 + w2);//32768/165375,  16384/165375, 8192/165375
                    float  g6 = 0.049535904f * (4*w0 - 2*w1 + w2);//32768/165375, -16384/165375, 8192/165375
                    
                    float  g7 = -2.8542885E-7f * (w0 + 3*w1 + 9*w2);//-1/3503500, -3/3503500, -9/3503500
                    float  g8 = -2.8542885E-7f * (w0 - 3*w1 + 9*w2);//-1/3503500,  3/3503500, -9/3503500
                    
                    float  g9 = -0.15168859f * (9*w0 + 3*w1 + w2);//-4782969/3503500, -1594323/3503500, -531441/3503500
                    float g10 = -0.15168859f * (9*w0 - 3*w1 + w2);//-4782969/3503500,  1594323/3503500, -531441/3503500
                    
                    float g11 = 6.2184937E-9f * (w0 + 4*w1 + 16*w2);//1/160810650,  2/80405325,  8/80405325
                    float g12 = 6.2184937E-9f * (w0 - 4*w1 + 16*w2);//1/160810650, -2/80405325,  8/80405325
                    
                    float g13 = 0.10432901f * (16*w0 + 4*w1 + w2);//134217728/80405325,  33554432/80405325, 8388608/80405325
                    float g14 = 0.10432901f * (16*w0 - 4*w1 + w2);//134217728/80405325, -33554432/80405325, 8388608/80405325
                    float g15 = w2;
                    
                    g[ 0][0] =  g0; 
                    g[ 1][0] =  g1; g[ 2][0] =  g2;
                    g[ 3][0] =  g3; g[ 4][0] =  g4; 
                    g[ 5][0] =  g5; g[ 6][0] =  g6;
                    g[ 7][0] =  g7; g[ 8][0] =  g8;
                    g[ 9][0] =  g9; g[10][0] = g10; 
                    g[11][0] = g11; g[12][0] = g12;
                    g[13][0] = g13; g[14][0] = g14; 
                    g[15][0] = g15;
                }
               
                //X transform---------------------------------------------------
                int ih = oh - ph + fh;
                int iw = ow - pw;
                
                float  x0 = ld(X, n, ih, iw    , ic, IH, IW);
                float  x1 = ld(X, n, ih, iw +  1, ic, IH, IW);
                float  x2 = ld(X, n, ih, iw +  2, ic, IH, IW);
                float  x3 = ld(X, n, ih, iw +  3, ic, IH, IW);
                float  x4 = ld(X, n, ih, iw +  4, ic, IH, IW);
                float  x5 = ld(X, n, ih, iw +  5, ic, IH, IW);
                float  x6 = ld(X, n, ih, iw +  6, ic, IH, IW);
                float  x7 = ld(X, n, ih, iw +  7, ic, IH, IW);
                float  x8 = ld(X, n, ih, iw +  8, ic, IH, IW);
                float  x9 = ld(X, n, ih, iw +  9, ic, IH, IW);
                float x10 = ld(X, n, ih, iw + 10, ic, IH, IW);
                float x11 = ld(X, n, ih, iw + 11, ic, IH, IW);
                float x12 = ld(X, n, ih, iw + 12, ic, IH, IW);
                float x13 = ld(X, n, ih, iw + 13, ic, IH, IW);
                float x14 = ld(X, n, ih, iw + 14, ic, IH, IW);
                float x15 = ld(X, n, ih, iw + 15, ic, IH, IW);
              
                float[][] d = new float[16][1]; {
                    float d0 = x0 - 30.42361f * x2 + 285.7587f * x4 - 937.158f * x6 + 937.158f * x8 - 285.7587f * x10 + 30.42361f * x12 - x14;
                    
                    float d1 =  x1 + x2 - 29.42361f * x3 - 29.42361f * x4 + 256.3351f * x5 + 256.3351f * x6 - 680.8229f * x7 - 680.8229f * x8 + 256.3351f * x9 + 256.3351f * x10 - 29.42361f * x11 - 29.42361f * x12 + x13 + x14;
                    float d2 = -x1 + x2 + 29.42361f * x3 - 29.42361f * x4 - 256.3351f * x5 + 256.3351f * x6 + 680.8229f * x7 - 680.8229f * x8 - 256.3351f * x9 + 256.3351f * x10 + 29.42361f * x11 - 29.42361f * x12 - x13 + x14;
                    
                    float d3 =  0.5f * x1 + 0.25f * x2 - 15.08681f * x3 - 7.543403f * x4 + 139.1076f * x5 + 69.55382f * x6 - 433.8021f * x7 - 216.901f * x8 + 360.1285f * x9 + 180.0642f * x10 - 52.84722f * x11 - 26.42361f * x12 + 2.0f * x13 + x14;
                    float d4 = -0.5f * x1 + 0.25f * x2 + 15.08681f * x3 - 7.543403f * x4 - 139.1076f * x5 + 69.55382f * x6 + 433.8021f * x7 - 216.901f * x8 - 360.1285f * x9 + 180.0642f * x10 + 52.84722f * x11 - 26.42361f * x12 - 2.0f * x13 + x14;
                    
                    float d5 =  2.0f * x1 + 4.0f * x2 - 52.84722f * x3 - 105.6944f * x4 + 360.1285f * x5 + 720.2569f * x6 - 433.8021f * x7 - 867.6042f * x8 + 139.1076f * x9 + 278.2153f * x10 - 15.08681f * x11 - 30.17361f * x12 + 0.5f * x13 + x14;
                    float d6 = -2.0f * x1 + 4.0f * x2 + 52.84722f * x3 - 105.6944f * x4 - 360.1285f * x5 + 720.2569f * x6 + 433.8021f * x7 - 867.6042f * x8 - 139.1076f * x9 + 278.2153f * x10 + 15.08681f * x11 - 30.17361f * x12 - 0.5f * x13 + x14;
                    
                    float d7 =  0.3333333f * x1 + 0.1111111f * x2 - 10.10417f * x3 - 3.368056f * x4 + 94.13021f * x5 + 31.37674f * x6 - 301.9271f * x7 - 100.6424f * x8 + 278.8385f * x9 + 92.94618f * x10 - 64.27083f * x11 - 21.42361f * x12 + 3.0f * x13 + x14;
                    float d8 = -0.3333333f * x1 + 0.1111111f * x2 + 10.10417f * x3 - 3.368056f * x4 - 94.13021f * x5 + 31.37674f * x6 + 301.9271f * x7 - 100.6424f * x8 - 278.8385f * x9 + 92.94618f * x10 + 64.27083f * x11 - 21.42361f * x12 - 3.0f * x13 + x14;
                    
                    float  d9 =  3.0f * x1 + 9.0f * x2 - 64.27083f * x3 - 192.8125f * x4 + 278.8385f * x5 + 836.5156f * x6 - 301.9271f * x7 - 905.7812f * x8 + 94.13021f * x9 + 282.3906f * x10 - 10.10417f * x11 - 30.3125f * x12 + 0.3333333f * x13 + x14;
                    float d10 = -3.0f * x1 + 9.0f * x2 + 64.27083f * x3 - 192.8125f * x4 - 278.8385f * x5 + 836.5156f * x6 + 301.9271f * x7 - 905.7812f * x8 - 94.13021f * x9 + 282.3906f * x10 + 10.10417f * x11 - 30.3125f * x12 - 0.3333333f * x13 + x14;

                    float d11 =  0.25f * x1 + 0.0625f * x2 - 7.590278f * x3 - 1.897569f * x4 + 70.96528f * x5 + 17.74132f * x6 - 229.8542f * x7 - 57.46354f * x8 + 219.9236f * x9 + 54.9809f * x10 - 57.69444f * x11 - 14.42361f * x12 + 4.0f * x13 + x14;
                    float d12 = -0.25f * x1 + 0.0625f * x2 + 7.590278f * x3 - 1.897569f * x4 - 70.96528f * x5 + 17.74132f * x6 + 229.8542f * x7 - 57.46354f * x8 - 219.9236f * x9 + 54.9809f * x10 + 57.69444f * x11 - 14.42361f * x12 - 4.0f * x13 + x14;
                    
                    float d13 =  4.0f * x1 + 16.0f * x2 - 57.69444f * x3 - 230.7778f * x4 + 219.9236f * x5 + 879.6944f * x6 - 229.8542f * x7 - 919.4167f * x8 + 70.96528f * x9 + 283.8611f * x10 - 7.590278f * x11 - 30.36111f * x12 + 0.25f * x13 + x14;
                    float d14 = -4.0f * x1 + 16.0f * x2 + 57.69444f * x3 - 230.7778f * x4 - 219.9236f * x5 + 879.6944f * x6 + 229.8542f * x7 - 919.4167f * x8 - 70.96528f * x9 + 283.8611f * x10 + 7.590278f * x11 - 30.36111f * x12 - 0.25f * x13 + x14;
                    
                    float d15 = -x1 + 30.42361f * x3 - 285.7587f * x5 + 937.158f * x7 - 937.158f * x9 + 285.7587f * x11 - 30.42361f * x13 + x15;
                    
                    d[0][0] = d0;
                    d[1][0] = d1; d[2][0] = d2;
                    d[3][0] = d3; d[4][0] = d4;
                    d[5][0] = d5; d[6][0] = d6;
                    d[7][0] = d7; d[8][0] = d8;
                    d[9][0] = d9; d[10][0] = d10;
                    d[11][0] = d11; d[12][0] = d12;
                    d[13][0] = d13; d[14][0] = d14;
                    d[15][0] = d15;
                }
                
                //accumulate----------------------------------------------------
                for(int t=0; t<16; t++) 
                    a[t][0] += g[t][0] * d[t][0];
            }
            
            float[][] y = new float[14][1];
            matMul(y, AT, a);//AT * accu: accu(16) -> Y(14)
            for(int t=0; t<14; t++) 
                wt(Y, y[t][0], n, oh, ow + t, oc, OW);
        }
        
        
       
    }
    
    static void println(float[] a, String name) {
        for(int i=0; i<a.length; i++) {
            float v = a[i];
            if(v == 0.0f) continue;
            
            //sign
            if(a[i] >= 0) System.out.print(" + ");
            else System.out.print(" - ");
            
            //value
            float abs = Math.abs(v);
            if(abs != 1.0f) System.out.print(abs + "f * ");
            
            System.out.print(name +  i);
        }
    }
   
    public static void main(String[] args){
        for(int i=0; i<BT.length; i++) {
            println(BT[i], "x");
            System.out.println();
            
        }
    }
}
