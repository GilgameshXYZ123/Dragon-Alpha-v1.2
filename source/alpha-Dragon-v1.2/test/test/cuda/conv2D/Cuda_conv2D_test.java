package test.cuda.conv2D;


import static z.dragon.alpha.Alpha.alpha;
import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
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
public class Cuda_conv2D_test 
{
    static { alpha.home("C:\\Users\\Gilgamesh\\Desktop\\Dragon-alpha-v1.2");}
    static final Mempool memp = alpha.engine.memp1(alpha.MEM_1GB * 8);
    static Engine eg = alpha.engine.cuda_float32(0, memp, alpha.MEM_1MB * 2048);
     
    public static void testCorrect(
            int IW, int FW, 
            int N, int IC, int OC,
            int sw, int pw)
    {
        int OW = (IW + (pw << 1) - FW) / sw + 1;
        
        System.out.println("Test correct:");
        System.out.format("\tXsize( N, IC, IW) = (%d, %d, %d)\n", N, IC, IW);
        System.out.format("\tWsize(OC, IC, FW) = (%d, %d, %d)\n", OC, IC,FW);
        System.out.format("\tYsize( N, OC, OW) = (%d, %d, %d)\n", N, OC, OW);
        
        int sizeX = N*IC*IW;
	int sizeW = OC*IC*FW;
	int sizeY = N*OC*OW;
        
        float[] X = Vector.random_float_vector(sizeX);
        float[] W = Vector.random_float_vector(sizeW);
        
        Tensor tX = eg.tensor(X, N, IW, IC);
        Tensor tW = eg.tensor(W, OC, FW, IC);
        
        //CPU-------------------------------------------------------------------
        float[][][] cX = Vector.to3D(X, N, IW, IC);
        float[][][] cW = Vector.to3D(W, OC, FW, IC);
        float[][][] cY = new float[N][OW][OC];
        Tense.conv2D_naive(cX, IW, cW, FW, cY, OW, N, IC, OC, sw, pw);
        float[] Y1 = Vector.flatten(cY);
                
        //GPU-------------------------------------------------------------------
        Tensor tY1 = eg.conv2D(tX, tW, sw, pw);
        Tensor tY2 = eg.conv2D(eg.empty(N, OW, OC).c(), tX, tW, sw);

        float[] Y2 = eg.valueOf(tY1);
        float[] Y3 = eg.valueOf(tY2);
        
        System.out.println(tY1);
        System.out.println(tY2);
        
        //compare----------------------------------------------------------------
        float sp1 = Vector.samePercent_relative(Y1, Y2, 1e-5f); 
        float sp2 = Vector.samePercent_relative(Y1, Y3, 1e-5f);
      
        System.out.print("CPU : "); Vector.println(Y1, 0, 10);
        System.out.print("GPU1: "); Vector.println(Y2, 0, 10);
        System.out.print("GPU2: "); Vector.println(Y3, 0, 10);

        System.out.println("sp1: " + sp1);
        System.out.println("sp2: " + sp2);
         
        eg.delete(tX, tW, tY1, tY2);
        if(sp1 <0.999f || sp2 < 0.999f) throw new RuntimeException();
        System.gc();
    }
    
    public static void main(String[] args)  {
        //test W11==============================================================
        {
            int IW = 32;
            int FW = 1;
            int N = 4;
            int IC = 128;
            int sw = 1, pw = 0;
            for(int oc = 1; oc <= 128; oc++) 
                testCorrect(IW, FW, N, IC, oc, sw, pw);
        }
    
        //test np===============================================================
        {
            int IW = 33;//(33 - 4 + 2)/1 + 1
            int FW = 5;//FH = 4, FW = 4
            int N = 4, IC = 16;
            int sw = 2, pw = 0;
            for (int oc = 1; oc <= 128; oc++) 
                testCorrect(IW, FW, N, IC, oc, sw, pw);
        }
        
        //test common===========================================================
        {
            int IW = 16;//(33 - 4 + 2)/1 + 1
            int FW = 3;
            int N = 4, IC = 16;
            int sw = 2, pw = 1;
            for(int oc = 1; oc <= 128; oc++) 
                testCorrect(IW, FW, N, IC, oc, sw, pw);
        }
        
        {
            int IW = 16;//(33 - 4 + 2)/1 + 1
            int FW = 4;
            int N = 4, IC = 16;
            int sw = 2, pw = 1;
            for(int oc = 1; oc <= 128; oc++) 
                testCorrect(IW, FW, N, IC, oc, sw, pw);
        }
        
        {
            int IW = 16;//(33 - 4 + 2)/1 + 1
            int FW = 5;
            int N = 4, IC = 16;
            int sw = 2, pw = 2;
            for(int oc = 1; oc <= 128; oc++)
                testCorrect(IW, FW, N, IC, oc, sw, pw);
        }
        
        {
            int IW = 8;//(33 - 4 + 2)/1 + 1
            int FW = 5;
            int N = 64, IC = 16;
            int sw = 2, pw = 2;
            for(int oc = 1; oc <= 128; oc++)
                testCorrect(IW, FW, N, IC, oc, sw, pw);
            
            sw = 1;
            for(int oc = 1; oc <= 128; oc++)
                testCorrect(IW, FW, N, IC, oc, sw, pw);
        }
        
        {
            int IW = 8;//(33 - 4 + 2)/1 + 1
            int FW = 6;
            int sw = 2, pw = 2;
            int N = 64;
            int IC = 16;
            for(int oc = 1; oc <= 128; oc++)
                testCorrect(IW, FW, N, IC, oc, sw, pw);
            
            sw = 1;
            for(int oc = 1; oc <= 128; oc++)
                testCorrect(IW, FW, N, IC, oc, sw, pw);
        }
        
        {
            int IW = 8;//(33 - 4 + 2)/1 + 1
            int FW = 5;
            int N = 64, IC = 16;
            int sw = 2, pw = 1;
            
            for(int oc = 1; oc <= 128; oc++)
                testCorrect(IW, FW, N, IC, oc, sw, pw);
            
            sw = 1;
            for(int oc = 1; oc <= 128; oc++) 
                testCorrect(IW, FW, N, IC, oc, sw, pw);
        }
        
        {
            int IW = 8;//(33 - 4 + 2)/1 + 1
            int FW = 7;
            int sw = 2, pw = 3;
            
            int N = 64;
            int IC = 16, OC = 128;//9*4=36 
                
            for(int oc = 1; oc <= 128; oc++)
                testCorrect(IW, FW, N, IC, oc, sw, pw);
            
            sw = 1;
            for(int oc = 1; oc <= 128; oc++)
                testCorrect(IW, FW, N, IC, oc, sw, pw);
        }
    }
}
