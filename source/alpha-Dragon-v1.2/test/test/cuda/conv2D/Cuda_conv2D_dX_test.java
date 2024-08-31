package test.cuda.conv2D;


import static z.dragon.alpha.Alpha.alpha;
import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.dragon.engine.cuda.impl.CudaException;
import z.dragon.engine.memp.Mempool;
import z.util.math.vector.Tense;
import z.util.math.vector.Vector;

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/**
 *
 * @author Gilgamesh
 */
public class Cuda_conv2D_dX_test 
{
    static { alpha.home("C:\\Users\\Gilgamesh\\Desktop\\Dragon-alpha-v1.2");}
    static final Mempool memp = alpha.engine.memp1(alpha.MEM_1GB * 8);
    static Engine eg = alpha.engine.cuda_float32(0, memp, alpha.MEM_1MB * 2048);
    
    public static void testCorrect(int IW,
            int OW, int FW,
            int N, int IC, int OC,
            int sw, int pw)
    {
        if(IW == -1) IW = (OW - 1)*sw + FW - 2*pw;
        int OWp = OW + (OW-1)*(sw-1);
        int opw = FW - pw - 1;
        
        System.out.println("Test correct:");
        System.out.format("\tYsize( N, OC, OW) = (%d, %d, %d)\n", N, OC, OW);
        System.out.format("\tXsize( N, IC, IW) = (%d, %d, %d)\n", N, IC, IW);
        System.out.format("\tWsize(OC, IC,FW) = (%d, %d, %d)\n", OC, IC, FW);
        System.out.format("\t(sw, pw) = (%d, %d)\n", sw, pw);
        System.out.format("\t(OW_p) = (%d)\n", OWp);
        System.out.format("\t(opw) = (%d)\n", opw);
        
        int sizeX = N  * IW * IC;
        int sizeW = OC * FW * IC;
	int sizeY = N  * OW * OC;
        
        float[] W = Vector.random_float_vector(sizeW);
        float[] deltaY = Vector.random_float_vector(sizeY);
        
        //CPU-------------------------------------------------------------------
        float[][][] cdeltaY = Vector.to3D(deltaY, N, OW, OC);
        float[][][] cW = Vector.to3D(W, OC, FW, IC);
        float[][][] cdeltaX = new float[N][IW][IC];
        
        Tense.deconv2D_deltaX_img2col(
                cdeltaY, OW, 
                cW, FW,
                cdeltaX, IW,
                N, IC, OC,
                sw, pw);
        
        float[] deltaX1 = Vector.flatten(cdeltaX);
        System.out.print("CPU1: "); Vector.println(deltaX1, 0, 10);
        
        //GPU-------------------------------------------------------------------
        Tensor tdeltaY = eg.tensor(deltaY, N, OW, OC);
        Tensor tW = eg.tensor(W, OC, FW, IC);
        
        Tensor tdeltaX1 = eg.conv2D_deltaX(tdeltaY, tW, IW, sw, pw).c();
        Tensor tdeltaX2 = eg.conv2D_deltaX(eg.empty(N, IW, IC), tdeltaY, tW, sw, pw).c();
        
        System.out.println(CudaException.lastException());
        
        //----------------------------------------------------------------------
        float[] deltaX2 = eg.valueOf(tdeltaX1);
        System.out.print("GPU : "); Vector.println(deltaX2, 0, 10);
        
        float[] deltaX3 = eg.valueOf(tdeltaX2);
        System.out.print("GPU : "); Vector.println(deltaX3, 0, 10);
        
        float sp1 = Vector.samePercent_relative(deltaX1, deltaX2, 1e-4f); System.out.println("sp1: "+sp1);
        float sp2 = Vector.samePercent_relative(deltaX1, deltaX3, 1e-4f); System.out.println("sp2: "+sp2);
        float zp1 = Vector.zeroPercent(deltaX2); System.out.println("zp1: " + zp1);
        float zp2 = Vector.zeroPercent(deltaX3); System.out.println("zp2: " + zp2);
        
        eg.delete(tdeltaY, tW, tdeltaX1);
        if(sp1 < 0.999f) throw new RuntimeException();
        if(sp2 < 0.999f) throw new RuntimeException();
    }
    
    public static void main(String[] args)
    {
        //test for W1===========================================================
        {
            int OW = 15;
            int FW = 1;
            int N = 4;
            int IC = 192, OC = 32;//128+64+32+16+8+4
            int sw = 1, pw = 0;
            for (int ic = 1; ic <= 128; ic++)
                testCorrect(-1, OW, FW, N, ic, OC, sw, pw);
        }
        
        //test for s1===========================================================
        {
            int OW = 8;
            int FW = 3;
            int N = 64, OC = 16;
            int sw = 1, pw = 1;

            for (int ic = 1; ic <= 128; ic++) 
                testCorrect(-1, OW, FW, N, ic, OC, sw, pw);
            
            sw = 2;
            for (int ic = 1; ic <= 128; ic++) 
                testCorrect(-1, OW, FW, N, ic, OC, sw, pw);
            
            sw = 3;
            for (int ic = 1; ic <= 128; ic++) 
                testCorrect(-1, OW, FW, N, ic, OC, sw, pw);
        }
        
        {
            int OW = 16, N = 4;
            int FW = 3;
            int OC = 16;
            int sw = 1, pw = 1;

            for (int ic = 1; ic <= 128; ic++) 
                testCorrect(-1, OW, FW, N, ic, OC, sw, pw);
        }
        
        //test ImsR-------------------------------------------------------------
        {
            int IW = 14, OW = 7;
            int FW = 3;
            int N = 64, OC = 32;
            int sw = 2, pw = 1;
            
            for (int ic = 64; ic <= 128; ic++) 
                testCorrect(IW, OW, FW, N, ic, OC, sw, pw);
        }
        
        {
            int IW = 22, OW = 7;
            int FW = 3;
            int N = 64, OC = 32;
            int sw = 3, pw = 1;
            for (int ic = 64; ic <= 128; ic++) 
                testCorrect(IW, OW, FW, N, ic, OC, sw, pw);
        }
        
        {
            int IW = 21, OW = 6;
            int FW = 3;
            int N = 64, OC = 32;
            int sw = 3, pw = 1;
            for (int ic = 64; ic <= 128; ic++) 
                testCorrect(IW, OW, FW, N, ic, OC, sw, pw);
        }
    }
}
