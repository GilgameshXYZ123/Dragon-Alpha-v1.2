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
public class Winograd_f14x3_v7 
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
            float  a0 = 0.0f,  a1 = 0.0f,  a2 = 0.0f,  a3 = 0.0f;
            float  a4 = 0.0f,  a5 = 0.0f,  a6 = 0.0f,  a7 = 0.0f;
            float  a8 = 0.0f,  a9 = 0.0f, a10 = 0.0f, a11 = 0.0f;
            float a12 = 0.0f, a13 = 0.0f, a14 = 0.0f, a15 = 0.0f;
            
            for(int fh=0; fh<3; fh++)
            for(int ic=0; ic<IC; ic++)
            {
                float t1, t2;

                //W transform---------------------------------------------------
                float w0 = W[oc][fh][0][ic];
                float w1 = W[oc][fh][1][ic];
                float w2 = W[oc][fh][2][ic];
               
                float g0 = w0;  //1, 0,  0,
                
                //B: g1, g2, g5, g6, g9, g10, g13, g14, 0, 15
                //S: g3, g4, g7, g8, g11, g12, 
                
                t1 = -2.222222e-03f*w0 - 2.222222e-03f*w2;
                t2 = -2.222222e-03f*w1;
                float g1 = t1 + t2, g2 = t1 - t2;

                t1 = 1.2093727E-5f*w0 + 4.8374906E-5f*w2;
                t2 = 2.4187453E-5f*w1;
                float g3 = t1 + t2, g4 = t1 - t2;
                     
                t1 = 0.19814362f*w0 + 0.049535904f*w2;
                t2 = 0.09907181f*w1;
                float g5 = t1 + t2, g6 = t1 - t2;
                
                t1 = -2.8542885E-7f*w0 - 2.5688596E-6f*w2;
                t2 = -8.562866E-7f*w1;
                float g7 = t1 + t2, g8 = t1 - t2;

                t1 = -1.3651974f*w0 - 0.15168859f*w2;
                t2 = -0.4550658f*w1;
                float g9 = t1 + t2, g10 = t1 - t2;

                t1 = 6.2184937E-9f*w0 + 9.94959E-8f*w2;
                t2 = 2.4873975E-8f*w1;
                float g11 = t1 + t2, g12 = t1 - t2;

                t1 = 1.6692642f*w0 + 0.10432901f*w2;
                t2 = 0.41731605f*w1;
                float g13 = t1 + t2, g14 = t1 - t2;
                
                float g15 = w2;
                    
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
                           
                float d0 = (x0 - x14) + (30.42361f * x12 - 30.42361f*x2) + (285.7587f*x4  - 285.7587f*x10)  + (937.158f*x8 - 937.158f*x6) ;
                
                t1 = (x1 + x13) + (256.3351f*x5  - (29.42361f*x3 + 29.42361f*x11)) + (256.3351f* x9 - 680.8229f*x7);
                t2 = (x2 + x14) + (256.3351f*x6  - (29.42361f*x4 + 29.42361f*x12)) + (256.3351f*x10 - 680.8229f*x8);
                float d1 = t2 + t1, d2 = t2 - t1;
                
                t1 = (0.5f *x1 + 2.0f*x13) + ((139.1076f*x5 - 52.84722f*x11) - 15.08681f*x3) + (360.1285f* x9 - 433.8021f*x7);
                t2 = (0.25f*x2 +      x14) + ((69.55382f*x6 - 26.42361f*x12) - 7.543403f*x4) + (180.0642f*x10 - 216.9010f*x8);
                float d3 = t2 + t1, d4 = t2 - t1;
                
                t1 = (2.0f*x1 + 0.5f*x13) + ((139.1076f*x9  - 52.84722f*x3) - 15.08681f*x11) + (360.1285f*x5 - 433.8021f*x7);
                t2 = (4.0f*x2 +      x14) + ((278.2153f*x10 - 105.6944f*x4) - 30.17361f*x12) + (720.2569f*x6 - 867.6042f*x8);
                float d5 = t2 + t1, d6 = t2 - t1;
                
                t1 = (0.3333333f*x1 + 3.0f*x13) - 10.10417f*x3 + (94.13021f*x5 - 64.27083f*x11) + (278.8385f* x9 - 301.9271f*x7);
                t2 = (0.1111111f*x2 +      x14) - 3.368056f*x4 + (31.37674f*x6 - 21.42361f*x12) + (92.94618f*x10 - 100.6424f*x8) ;
                float d7 = t2 + t1, d8 = t2 - t1;
                
                t1 = 3.0f*x1 - 64.27083f*x3 + 278.8385f*x5 - 301.9271f*x7 + 94.13021f* x9 - 10.10417f*x11 + 0.3333333f * x13;
                t2 = 9.0f*x2 - 192.8125f*x4 + 836.5156f*x6 - 905.7812f*x8 + 282.3906f*x10 - 30.31250f*x12 +              x14;
                float d9 = t2 + t1, d10 = t2 - t1;
                
                t1 = 0.25f  *x1 + ((4.0f*x13 - 7.590278f*x3) - (57.69444f*x11 - 70.96528f*x5)) - (229.8542f*x7 - 219.9236f* x9);
                t2 = 0.0625f*x2 + ((     x14 - 1.897569f*x4) - (14.42361f*x12 - 17.74132f*x6)) - (57.46354f*x8 - 54.98090f*x10);
                float d11 = t2 + t1, d12 = t2 - t1;
               
                t1 =  4.0f*x1 - 7.590278f*x11 - 57.69444f*x3 + 70.96528f* x9 + 219.9236f*x5 - 229.8542f*x7 + 0.25f * x13;
                t2 = 16.0f*x2 - 30.36111f*x12 - 230.7778f*x4 + 283.8611f*x10 + 879.6944f*x6 - 919.4167f*x8  +         x14;
                float d13 = t2 + t1, d14 = t2 - t1;
                        
                float d15 =  (x15 - x1) + (30.42361f*x3  - 30.42361f*x13) + (285.7587f*x11 - 285.7587f*x5) + (937.158f*x7 - 937.158f*x9);
                
                //accumulate----------------------------------------------------
                 a0 +=  g0 *  d0;  a1 +=  g1 *  d1;  a2 +=  g2 *  d2;  a3 +=  g3 *  d3;
                 a4 +=  g4 *  d4;  a5 +=  g5 *  d5;  a6 +=  g6 *  d6;  a7 +=  g7 *  d7;
                 a8 +=  g8 *  d8;  a9 +=  g9 *  d9; a10 += g10 * d10; a11 += g11 * d11;
                a12 += g12 * d12; a13 += g13 * d13; a14 += g14 * d14; a15 += g15 * d15;
            }
            
            float y0 = a0 + (a1 + a2) + (a3 + a4) + (a5 + a6) + (a7 + a8) + (a9 + a10) + (a11 + a12) + (a13 + a14);
            
          //B: g1, g2, g5, g6, g9, g10, g13, g14, 0, 15
          //S: g3, g4, g7, g8, g11, g12, 
            
            float y1 = ((a1 - a2) + 0.5f  * (a5 - a6) + 0.3333333f * (a9 - a10) + 0.25f   * (a13 - a14)) + (2.0f * (a3 - a4) + 3.0f * (a7 - a8) +  4.0f * (a11 - a12));
            float y2 = ((a1 + a2) + 0.25f * (a5 + a6) + 0.1111111f * (a9 + a10) + 0.0625f * (a13 + a14)) + (4.0f * (a3 + a4) + 9.0f * (a7 + a8) + 16.0f * (a11 + a12));
            
            float y3 = ((a1 - a2) + 0.125f  * (a5 - a6)  + 0.03703704f * (a9 - a10)  + 0.015625f  * (a13 - a14)) + ( 8.0f * (a3 - a4) + 27.0f * (a7 - a8) +  64.0f * (a11 - a12));
            float y4 = ((a1 + a2) + 0.0625f * (a5 + a6)  + 0.01234568f * (a9 + a10) + 0.00390625f * (a13 + a14)) + (16.0f * (a3 + a4) + 81.0f * (a7 + a8) + 256.0f * (a11 + a12)) ;
            
            float y5 = ((a1 - a2) + 0.03125f  * (a5 - a6) + 32.0f * (a3 - a4) + 0.004115226f * (a9 - a10) + 9.765625E-4f * (a13 - a14)) + 243.0f * (a7 - a8) + 1024.0f * (a11 - a12);
            float y6 = ((a1 + a2) + 0.015625f * (a5 + a6) + 64.0f * (a3 + a4) + 0.001371742f * (a9 + a10) + 2.441406E-4f * (a13 + a14)) + 729.0f * (a7 + a8) + 4096.0f * (a11 + a12);
            
            float y7 = ((a1 - a2) + 0.0078125f  * (a5 - a6) + 4.572474E-4f * (a9 - a10) + 6.103516E-5f * (a13 - a14)) + (128.0f * (a3 - a4) + 2187.0f * (a7 - a8) + 16384.0f * (a11 - a12));
            float y8 = ((a1 + a2) + 0.00390625f * (a5 + a6) + 1.524158E-4f * (a9 + a10) + 1.525879E-5f * (a13 + a14)) + (256.0f * (a3 + a4) + 6561.0f * (a7 + a8) + 65536.0f * (a11 + a12));

            float  y9 = ((a1 - a2) + 0.001953125f * (a5 - a6) + 5.080526E-5f * (a9 - a10) + 3.814697E-6f * (a13 - a14)) + ( 512.0f * (a3 - a4) + 19683.0f * (a7 - a8) +  262144.0f * (a11 - a12));
            float y10 = ((a1 + a2) + 9.765625E-4f * (a5 + a6) + 1.693509E-5f * (a9 + a10) + 9.536743E-7f * (a13 + a14)) + (1024.0f * (a3 + a4) + 59049.0f * (a7 + a8) + 1048576.0f * (a11 + a12));
            
            float y11 = ((a1 - a2) + 4.882812E-4f * (a5 - a6) + 5.645029E-6f * (a9 - a10) + 2.384186E-7f * (a13 - a14)) + (2048.0f * (a3 - a4) + 177147.0f * (a7 - a8) +  4194304.0f * (a11 - a12));
            float y12 = ((a1 + a2) + 2.441406E-4f * (a5 + a6) + 1.881676E-6f * (a9 + a10) + 5.960464E-8f * (a13 + a14)) + (4096.0f * (a3 + a4) + 531441.0f * (a7 + a8)  + 1.677722E7f * (a11 + a12));
            
            float y13 = (a1 - a2) + 8192.0f * (a3 - a4) + 1.220703E-4f * (a5 - a6) + 1594323.0f * (a7 - a8) + 6.272255E-7f * (a9 - a10) + 6.710886E7f * (a11 - a12) + 1.490116E-8f * (a13 - a14) + a15;
            
            wt(Y, y0, n, oh, ow     , oc, OW);
            wt(Y, y1, n, oh, ow +  1, oc, OW);
            wt(Y, y2, n, oh, ow +  2, oc, OW);
            wt(Y, y3, n, oh, ow +  3, oc, OW);
            wt(Y, y4, n, oh, ow +  4, oc, OW);
            wt(Y, y5, n, oh, ow +  5, oc, OW);
            wt(Y, y6, n, oh, ow +  6, oc, OW);//0.5f
            
            wt(Y,  y7, n, oh, ow +  7, oc, OW);
            wt(Y,  y8, n, oh, ow +  8, oc, OW);
            wt(Y,  y9, n, oh, ow +  9, oc, OW);
            wt(Y, y10, n, oh, ow + 10, oc, OW);
            wt(Y, y11, n, oh, ow + 11, oc, OW);
            wt(Y, y12, n, oh, ow + 12, oc, OW);
            wt(Y, y13, n, oh, ow + 13, oc, OW);
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
        for(int i=0; i<AT.length; i++) {
            System.out.print("float y" + i + " = ");
            println(AT[i], "a");
            System.out.println();
            
        }
    }
}
