package test.cuda.conv3D;


import static z.dragon.alpha.Alpha.alpha;
import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.dragon.engine.cuda.CudaFloat32EngineBase;
import z.dragon.engine.cuda.impl.Cuda;
import z.dragon.engine.cuda.impl.math.Cuda_conv3D;
import z.dragon.engine.memp.Mempool;
import z.util.lang.SimpleTimer;
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
public class Cuda_deconv3D_biased_test 
{
    static { alpha.home("C:\\Users\\Gilgamesh\\Desktop\\Dragon-alpha-v1.2");}
    static Mempool memp = alpha.engine.memp1();
    static Engine eg = alpha.engine.cuda_float32(0, memp);
    static {
        CudaFloat32EngineBase cu32 = (CudaFloat32EngineBase) eg.engineBase();
//        cu32.conv3D_remode(false);
//        cu32.conv3D_useTexture(false);
    }
     
     public static void testCorrect(
            int IH, int IW, 
            int FH, int FW, 
            int N, int IC, int OC,
            int sh, int sw, int ph, int pw)
    {
        eg.sync(false);
        
        int OH = (IH - 1)*sh + FH - (ph << 1);//floor
        int OW = (IW - 1)*sw + FW - (pw << 1);//floor
        
        System.out.println("Test correct:");
        System.out.format("\tXsize( N, IC, IH, IW) = (%d, %d, %d, %d)\n", N, IC, IH, IW);
        System.out.format("\tWsize(OC, IC, FH, FW) = (%d, %d, %d, %d)\n", OC, IC, FH, FW);
        System.out.format("\tYsize( N, OC, OH, OW) = (%d, %d, %d, %d)\n", N, OC, OH, OW);
        
        Tensor X = eg.Gaussian(N, IH, IW, IC).c();
        Tensor W = eg.Gaussian(IC, FH, FW, OC).c();
        Tensor B = eg.Gaussian(OC).c();
        
        Tensor Y1 = eg.deconv3D(eg.empty(N, OH, OW, OC).c(), X, W, sh, sw); eg.add_row(true, Y1.c(), B);
        Tensor Y2 = eg.deconv3D(X, W, OH, OW, sh, sw, ph, pw); eg.add_row(true, Y2.c(), B);
        
        Tensor Y3 = eg.deconv3D_biased(X, W, sh, sw, ph, pw, B);
        Tensor Y4 = eg.deconv3D_biased(eg.empty(N, OH, OW, OC).c(), X, W, sh, sw, ph, pw, B).c();
        
        float[] v1 = Y1.value(); 
        float[] v2 = Y2.value();
        float[] v3 = Y3.value();
        float[] v4 = Y4.value();
        float sp1 = Vector.samePercent_absolute(v1, v3);
        float sp2 = Vector.samePercent_absolute(v2, v4);
        
        Vector.println("v1 = ", v1, 0, 10);
        Vector.println("v2 = ", v2, 0, 10);
        Vector.println("v3 = ", v3, 0, 10);
        Vector.println("v4 = ", v4, 0, 10);
        
        System.out.println("sp1 = " + sp1);
        System.out.println("sp2 = " + sp2);
        
        if(sp1 <0.999f || sp2 < 0.999f) throw new RuntimeException();
        System.gc();
    }
     
    public static void main(String[] args)
    {
        {
            int IH = 16, IW = 16;//(33 - 4 + 2)/1 + 1
            int FH = 3, FW = 3;
            int sh = 1, sw = 1, ph = 1, pw = 1;
            int N = 4;
            int IC = 16, OC = 128;//9*4=36 
            for(int oc = 1; oc <= 128; oc++) {
                testCorrect(IH, IW, FH, FW, N, IC, oc, sh, sw, ph, pw);
            }
        }
        
        {
            int IH = 16, IW = 16;//(33 - 4 + 2)/1 + 1
            int FH = 4, FW = 4;
            int sh = 2, sw = 2, ph = 1, pw = 1;
            int N = 4;
            int IC = 16, OC = 128;//9*4=36 
            for(int oc = 1; oc <= 128; oc++) {
                testCorrect(IH, IW, FH, FW, N, IC, oc, sh, sw, ph, pw);
            }
        }
        
        {
            int IH = 16, IW = 16;//(33 - 4 + 2)/1 + 1
            int FH = 5, FW = 5;
            int sh = 2, sw = 2, ph = 2, pw = 2;
            int N = 4;
            int IC = 16, OC = 128;//9*4=36 
            for(int oc = 1; oc <= 128; oc++) {
                testCorrect(IH, IW, FH, FW, N, IC, oc, sh, sw, ph, pw);
            }
        }
        
        {
            int IH = 8, IW = 8;//(33 - 4 + 2)/1 + 1
            int FH = 5, FW = 5;
            int sh = 2, sw = 2, ph = 2, pw = 2;
            int N = 64;
            int IC = 16, OC = 128;//9*4=36 
            for(int oc = 1; oc <= 128; oc++) {
                testCorrect(IH, IW, FH, FW, N, IC, oc, sh, sw, ph, pw);
            }
        }
    }
}
