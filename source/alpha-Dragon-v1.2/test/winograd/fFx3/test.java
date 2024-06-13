/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package winograd.fFx3;

import z.util.math.vector.Tense;
import z.util.math.vector.Vector;

/**
 *
 * @author Gilgamesh
 */
public class test 
{
    public static void testCorrect(
            int IH, int IW, 
            int OH, int OW,
            int FH, int FW, 
            int N, int IC, int OC,
            int sh, int sw, int ph, int pw,
            float bound)
    {
        System.out.println("Test correct:");
        System.out.format("\tXsize( N, IC, IH, IW) = (%d, %d, %d, %d)\n", N, IC, IH, IW);
        System.out.format("\tWsize(OC, IC, FH, FW) = (%d, %d, %d, %d)\n", OC, IC, FH, FW);
        System.out.format("\tYsize( N, OC, OH, OW) = (%d, %d, %d, %d)\n", N, OC, OH, OW);
        
        int sizeX = N*IH*IW*IC;
	int sizeW = OC*IC*FH*FW;
	int sizeY = N*OC*OH*OW;
        
        float[] X = Vector.random_float_vector(sizeX);
        float[] W = Vector.random_float_vector(sizeW, 0, bound);
        
        //Direct----------------------------------------------------------------
        float[][][][] cX = Tense.vectorToTensor_4D(X, N, IH, IW, IC);
        float[][][][] cW = Tense.vectorToTensor_4D(W, OC, FH, FW, IC);
        float[][][][] cY1 = new float[N][OH][OW][OC];
        float[][][][] cY2 = new float[N][OH][OW][OC];
        
        Tense.conv3D_naive(
                cX, IH, IW,
                cW, FH, FW,
                cY1, OH, OW,
                N, IC, OC, 
                sh, sw, ph, pw);
        
        //Winograd--------------------------------------------------------------
        Winograd_f14x3_v6.winograd(
                cX, IH, IW,
                cW, FH, FW, 
                cY2, OH, OW, 
                N, IC, OC,
                ph, pw);
        
        //compare---------------------------------------------------------------
        float[] Y1 = Vector.flatten(cY1);
        float[] Y2 = Vector.flatten(cY2);
        float sp = Vector.samePercent_absolute(Y1, Y2, 0.01f);
        
        float df1 = Vector.difference_absolute(Y1, Y2);
        float df2 = Vector.difference_relative(Y1, Y2);
        
        float zp0 = Vector.zeroPercent(Y1);
        float zp1 = Vector.zeroPercent(Y2);
        
        Vector.println("Y1 = ", Y1, 0, 10);
        Vector.println("Y2 = ", Y2, 0, 10);
        System.out.println("sp = " + sp);
        System.out.println("zp0 = " + zp0);
        System.out.println("zp1 = " + zp1);
        
        System.out.println("difference1 = " + df1);
        System.out.println("difference2 = " + df2);
        
        if(sp < 0.9f) System.exit(-1111);
        if(sp < 0.99f) throw new RuntimeException();
    }
    
     public static void distribute(
            int IH, int IW, 
            int OH, int OW,
            int FH, int FW, 
            int N, int IC, int OC,
            int sh, int sw, int ph, int pw,
            float bound)
    {
        System.out.println("Test correct:");
        System.out.format("\tXsize( N, IC, IH, IW) = (%d, %d, %d, %d)\n", N, IC, IH, IW);
        System.out.format("\tWsize(OC, IC, FH, FW) = (%d, %d, %d, %d)\n", OC, IC, FH, FW);
        System.out.format("\tYsize( N, OC, OH, OW) = (%d, %d, %d, %d)\n", N, OC, OH, OW);
        
        int sizeX = N*IH*IW*IC;
	int sizeW = OC*IC*FH*FW;
	int sizeY = N*OC*OH*OW;
        
        float[] X = Vector.random_float_vector(sizeX);
        float[] W = Vector.random_float_vector(sizeW, 0, bound);
        
        //Direct----------------------------------------------------------------
        float[][][][] cX = Tense.vectorToTensor_4D(X, N, IH, IW, IC);
        float[][][][] cW = Tense.vectorToTensor_4D(W, OC, FH, FW, IC);
        float[][][][] cY1 = new float[N][OH][OW][OC];
        float[][][][] cY2 = new float[N][OH][OW][OC];
        
        
        //Winograd--------------------------------------------------------------
        float[] dis = new float[16];
        for(int i=0; i<10000; i++) {
            float[] v = Winograd_f14x3_v9.winograd(
                        cX, IH, IW,
                        cW, FH, FW, 
                        cY2, OH, OW, 
                        N, IC, OC,
                        ph, pw);
            for(int t=0; t<16; t++) dis[t] += v[t] / 1000;
        }
       
        Vector.println(dis);
        
        //compare---------------------------------------------------------------
        float[] Y1 = Vector.flatten(cY1);
        float[] Y2 = Vector.flatten(cY2);
        float sp = Vector.samePercent_absolute(Y1, Y2, 0.01f);
        
        float df1 = Vector.difference_absolute(Y1, Y2);
        float df2 = Vector.difference_relative(Y1, Y2);
        
        float zp0 = Vector.zeroPercent(Y1);
        float zp1 = Vector.zeroPercent(Y2);
        
        Vector.println("Y1 = ", Y1, 0, 10);
        Vector.println("Y2 = ", Y2, 0, 10);
        System.out.println("sp = " + sp);
        System.out.println("zp0 = " + zp0);
        System.out.println("zp1 = " + zp1);
        
        System.out.println("difference1 = " + df1);
        System.out.println("difference2 = " + df2);
        
        if(sp < 0.9f) System.exit(-1111);
        if(sp < 0.99f) throw new RuntimeException();
    }
    
    
    public static void main(String[] args)
    {
        System.out.println((float)(134217728.0/80405325));
        
        
//        int IH = 12, IW = 12, OH = 12, OW = 12;
        int IH = 14, IW = 14, OH = 14, OW = 14;
//        int IH = 16, IW = 16, OH = 16, OW = 16; 
	int FH = 3, FW = 3;//4*4*8 = 32*4 = 128
        int IC = 128, OC = 32;//9*4=36 3*3*3 = 9*3 = 27;
	int N = 4;
	int sh = 1, sw = 1, ph = 1, pw = 1;
        
//        Vector.PRINT_DIFFERENT = true;
           
//        distribute(IH, IW, OH, OW, FH, FW, N, IC, OC, sh, sw, ph, pw, 0.1f);
        
        testCorrect(IH, IW, OH, OW, FH, FW, N, IC, OC, sh, sw, ph, pw, 0.1f);
        testCorrect(IH, IW, OH, OW, FH, FW, N, IC/2, OC, sh, sw, ph, pw, 0.1f);
        testCorrect(IH, IW, OH, OW, FH, FW, N, IC*2, OC, sh, sw, ph, pw, 0.1f);
        testCorrect(IH, IW, OH, OW, FH, FW, N, IC*4, OC, sh, sw, ph, pw, 0.1f);
////        
//        for(int oc = 32; oc<=37; oc++)
//            testCorrect(IH, IW, OH, OW, FH, FW, N, IC, oc, sh, sw, ph, pw, 1);
    }
}
