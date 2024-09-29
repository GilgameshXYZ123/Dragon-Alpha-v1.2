package test.cuda.conv3D;


import static z.dragon.alpha.Alpha.alpha;
import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.dragon.engine.cuda.CudaFloat32EngineBase;
import z.dragon.engine.cuda.impl.Cuda;
import z.dragon.engine.cuda.impl.CudaException;
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
public class Cuda_conv3D_dX_test 
{
    static { alpha.home("C:\\Users\\Gilgamesh\\Desktop\\Dragon-alpha-v1.2");}
    static Mempool memp = alpha.engine.memp1(alpha.MEM_1GB * 8);
    static Engine eg = alpha.engine.cuda_float32(0, memp, alpha.MEM_1MB * 2048);
    static CudaFloat32EngineBase cu32;
    static {
        cu32 = (CudaFloat32EngineBase) eg.engineBase();
//        cu32.dconv3D_deltaX_s1_useTexture(false);
//        cu32.conv3D_deltaX_ks_useTexture(true);
    }
    
    public static void testCorrect(int IH, int IW,
            int OH, int OW,
            int FH, int FW,
            int N, int IC, int OC,
            int sh, int sw, int ph, int pw)
    {
        if(IH == -1) IH = (OH - 1)*sh + FH - 2*ph;
        if(IW == -1) IW = (OW - 1)*sw + FW - 2*pw;

        int OHp = OH + (OH-1)*(sh-1), OWp = OW + (OW-1)*(sw-1);
        int oph = FH - ph - 1, opw = FW - pw - 1;
        
        System.out.println("Test correct:");
        System.out.format("\tYsize( N, OC, OH, OW) = (%d, %d, %d, %d)\n", N, OC, OH, OW);
        System.out.format("\tXsize( N, IC, IH, IW) = (%d, %d, %d, %d)\n", N, IC, IH, IW);
        System.out.format("\tWsize(OC, IC, FH, FW) = (%d, %d, %d, %d)\n", OC, IC, FH, FW);
        System.out.format("\t(sh, sw, ph, pw) = (%d, %d, %d, %d)\n", sh, sw, ph, pw);
        System.out.format("\t(OH_p, OW_p) = (%d, %d)\n", OHp, OWp);
        System.out.format("\t(oph, opw) = (%d, %d)\n", oph, opw);
        
        int GN = IC;
        int GM = N  * IH * IW;
        int GK = OC * FH * FW;
        System.out.format("(GN, GM, GK) = (%d, %d, %d)\n", GN, GM, GK);
        
        int sizeX = N  * IH * IW * IC;
        int sizeW = OC * FH * FW * IC;
	int sizeY = N  * OH * OW * OC;
        
        float[] W = Vector.random_float_vector(sizeW);
        float[] deltaY = Vector.random_float_vector(sizeY);
        
        //CPU-------------------------------------------------------------------
        float[][][][] cdeltaY = Tense.vectorToTensor_4D(deltaY, N, OH, OW, OC);
        float[][][][] cW = Tense.vectorToTensor_4D(W, OC, FH, FW, IC);
        float[][][][] cdeltaX = new float[N][IH][IW][IC];
        
        Tense.deconv3D_deltaX_img2col(
                cdeltaY, OH, OW, 
                cW, FH, FW,
                cdeltaX, IH, IW,
                N, IC, OC,
                sh, sw, ph, pw);
        
        float[] deltaX1 = Tense.tensor_4DToVector(cdeltaX, sizeX); 
        System.out.print("CPU1: "); Vector.println(deltaX1, 0, 10);
        
        //GPU-------------------------------------------------------------------
        Tensor tdeltaY = eg.tensor(deltaY, N, OH, OW, OC);
        Tensor tW = eg.tensor(W, OC, FH, FW, IC);
        
        
        Tensor tdeltaX1 = eg.conv3D_deltaX(tdeltaY, tW, IH, IW, sh, sw, ph, pw).c();
        Tensor tdeltaX2 = eg.conv3D_deltaX(eg.empty(N, IH, IW, IC), tdeltaY, tW, sh, sw, ph, pw).c();
        
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
    
    public static void testSpeed(int IH, int IW,
            int OH, int OW,
            int FH, int FW,
            int N, int IC, int OC,
            int sh, int sw, int ph, int pw)
    {
        eg.check(false).sync(false);
        
        if(IH == -1) IH = (OH - 1)*sh + FH - 2*ph;
        if(IW == -1) IW = (OW - 1)*sw + FW - 2*pw;
        
        int OHp = OH + (OH-1)*(sh-1), OWp = OW + (OW-1)*(sw-1);
        int oph = FH - ph - 1, opw = FW - pw - 1;
        
        System.out.println("Test Speed:");
        System.out.format("\tYsize( N, OC, OH, OW) = (%d, %d, %d, %d)\n", N, OC, OH, OW);
        System.out.format("\tXsize( N, IC, IH, IW) = (%d, %d, %d, %d)\n", N, IC, IH, IW);
        System.out.format("\tWsize(OC, IC, FH, FW) = (%d, %d, %d, %d)\n", OC, IC, FH, FW);
        System.out.format("\t(sh, sw, ph, pw) = (%d, %d, %d, %d)\n", sh, sw, ph, pw);
        System.out.format("\t(OH_p, OW_p) = (%d, %d)\n", OHp, OWp);
        System.out.format("\t(oph, opw) = (%d, %d)\n", oph, opw);
        
        int GN = IC;
        int GM = N  * IH * IW;
        int GK = OC * FH * FW;
        System.out.format("(GN, GM, GK) = (%d, %d, %d)\n", GN, GM, GK);
        
        int sizeX = N  * IH * IW * IC;
        int sizeW = OC * FH * FW * IC;
	int sizeY = N  * OH * OW * OC;
        
        float[] W = Vector.random_float_vector(sizeW);
        float[] deltaY = Vector.random_float_vector(sizeY);
        
        Tensor tdeltaY = eg.tensor(deltaY, N, OH, OW, OC);
        Tensor tW = eg.tensor(W, OC, FH, FW, IC);
        Tensor tdeltaX = eg.empty(N, IH, IW, IC);
                
        tdeltaX = eg.conv3D_deltaX(tdeltaY, tW, IH, IW, sh, sw, ph, pw).c(); eg.delete(tdeltaX);//sync
        SimpleTimer timer = new SimpleTimer().record();
        int nIter = 500;
        for(int i=0; i<nIter; i++)
        {
//            cu32.conv3D_deltaX_decide_algorithm(OH, OW, IH, IW, FH, FW, N, IC, OC, sh, sw, ph, pw);
            tdeltaX = eg.conv3D_deltaX(tdeltaY, tW, IH, IW, sh, sw, ph, pw).c(); eg.delete(tdeltaX);//sync
        }
        Cuda.deviceSynchronize();
        
        long div = timer.record().timeStamp_dif_millis();
        float time = 1.0f*div/nIter;
        double sizeV = 1.0* GN/1024*GM/1024*GK/1024;
        float performance = (float)((sizeV*1024*1024 * 1e-9 * 2 * 1024)/(time*1e-3));
        System.out.format("Size = %f, Time = %f msec, Performance = %f GFlop/s\n", 
                (float)sizeV, time, performance);
        
        eg.delete(tdeltaX, tdeltaY, tW);
    }
    
    public static void main(String[] args)
    {
//        testCorrect(128, 128, 128, 128, 3, 3, 256,   3,  64, 1, 1, 1, 1);//correct: [256, 128, 128,   3] -> [256, 128, 128,  64]
//        testCorrect(128, 128, 128, 128, 3, 3, 256,  64,  64, 1, 1, 1, 1);//correct: [256, 128, 128,  64] -> [256, 128, 128,  64]
        
//        Vector.PRINT_DIFFERENT = true;
        {
            //int FH = 9, FW = 9, sh = 1, sw = 1, ph = 4, pw = 4;
//            int FH = 7, FW = 7, sh = 1, sw = 1, ph = 3, pw = 3;
            int FH = 8, FW = 8, sh = 1, sw = 1, ph = 4, pw = 4;
            
//           int FH = 3, FW = 3, sh = 1, sw = 1, ph = 1, pw = 1;
//           int IH = 128, IW = 128, IC =   3, OC =  64, N = 128;//false
//           int IH = 128, IW = 128, IC =   3, OC =  64, N = 128;//false

        
//            int IH = 64, IW = 64, OH = 64, OW = 64, N = 128, IC = 64, OC = 64;
//            int IH = 32, IW = 32, OH = 32, OW = 32, N = 64, IC = 128, OC = 128;
//            int IH = 16, IW = 16, OH = 16, OW = 16, N = 128, IC =  256, OC = 256;
//            int IH = 16, IW = 16, IC = 128, OC = 128, N = 128;
//            int IH =  8, IW =  8, IC = 128, OC = 256, N = 128;
//            int IH =  8, IW =  8, IC = 256, OC = 256, N = 128;

//            int IH = 16, IW = 16, OH = 16, OW = 16, N = 128, IC = 512, OC = 512;

//          //======[7 * 7]=====================================================
//            int IH = 128, IW = 128, OH = 128, OW = 128, N = 32, IC = 64, OC = 64;
//            int IH = 64, IW = 64, OH = 64, OW = 64, N = 64, IC = 64, OC = 64;
//            int IH = 32, IW = 32, OH = 32, OW = 32, N = 32, IC = 256, OC = 256;
//            int IH = 16, IW = 16, OH = 16, OW = 16, N = 32, IC = 512, OC = 512;
//            int IH = 120, IW = 120, OH = 120, OW = 120, N = 32, IC = 64, OC = 64;
//            int IH = 80, IW = 80, OH = 80, OW = 80, N = 64, IC = 64, OC = 64;
//            int IH = 40, IW = 40, OH = 40, OW = 40, N = 128, IC = 128, OC = 128;
//            int IH = 20, IW = 20, OH = 20, OW = 20, N = 64, IC = 256, OC = 256;
//            int IH = 10, IW = 10, OH = 10, OW = 10, N = 64, IC = 512, OC = 512;
//            System.out.println((IC & 63) == 0);

            //======[8 * 8]=======================================================
            //int IH = 72, IW = 72, OH = 72, OW = 72, N = 64, IC = 64, OC = 64;
            //int IH = 36, IW = 36, OH = 36, OW = 36, N = 64, IC = 128, OC = 128;
            //int IH = 18, IW = 18, OH = 18, OW = 18, N = 64, IC = 256, OC = 256;
            //int IH = 9, IW = 9, OH = 9, OW = 9, N = 64, IC = 512, OC = 512;
            //int IH = 64, IW = 64, OH = 64, OW = 64, N = 128, IC = 64, OC = 64;
            //int IH = 32, IW = 32, OH = 32, OW = 32, N = 128, IC = 128, OC = 128;
            //int IH = 16, IW = 16, OH = 16, OW = 16, N = 128, IC = 512, OC = 512;

            //int IH = 128, IW = 128, OH = 128, OW = 128, N = 32, IC = 64, OC = 64;
            int IH = 64, IW = 64, OH = 64, OW = 64, N = 128, IC = 64, OC = 64;
            //int IH = 112, IW = 112, OH = 112, OW = 112, N = 32, IC = 64, OC = 64;
            //int IH = 56, IW = 56, OH = 56, OW = 56, N = 128, IC = 64, OC = 64;
            //int IH = 28, IW = 28, OH = 28, OW = 28, N = 128, IC = 128, OC = 128;

//          testCorrect(IH, IW, FH, FW, N, IC, OC, sh, sw, ph, pw);
            testSpeed(IH, IW, OH, OW, FH, FW, N, IC, OC, sh, sw, ph, pw);
            System.exit(-2);
        }
        
        
        //test for W1===========================================================
        {
//            int OH = 31, OW = 31;
//            int OH = 15, OW = 15;
            int OH = 7, OW = 7;
            int FH = 1, FW = 1;
            int N = 4;
            int IC = 192, OC = 32;//128+64+32+16+8+4
            int sh = 1, sw = 1, ph = 0, pw = 0;
            for (int ic = 1; ic <= 128; ic++)
                testCorrect(-1, -1, OH, OW, FH, FW, N, ic, OC, sh, sw, ph, pw);
        }
        
        //test for s1===========================================================
        {
            int OH = 4, OW = 4, N = 64;
//            int FH = 5, FW = 5, ph = 2, pw = 2;
//            int FH = 4, FW = 4, ph = 1, pw = 1;
            int FH = 3, FW = 3, ph = 1, pw = 1;
            int OC = 16;//9*4=36 
            int sh = 1, sw = 1;

            for (int ic = 1; ic <= 128; ic++) 
                testCorrect(-1, -1, OH, OW, FH, FW, N, ic, OC, sh, sw, ph, pw);
            
            sh = 2; sw = 2;
            for (int ic = 1; ic <= 128; ic++) 
                testCorrect(-1, -1, OH, OW, FH, FW, N, ic, OC, sh, sw, ph, pw);
            
            sh = 3; sw = 3;
            for (int ic = 1; ic <= 128; ic++) 
                testCorrect(-1, -1, OH, OW, FH, FW, N, ic, OC, sh, sw, ph, pw);
        }
        
        {
            int OH = 16, OW = 16, N = 4;
//            int FH = 5, FW = 5, ph = 2, pw = 2;
//            int FH = 4, FW = 4, ph = 1, pw = 1;
            int FH = 3, FW = 3, ph = 1, pw = 1;
            int OC = 16;//9*4=36 
            int sh = 1, sw = 1;

            for (int ic = 1; ic <= 128; ic++) 
                testCorrect(-1, -1, OH, OW, FH, FW, N, ic, OC, sh, sw, ph, pw);
        }
        
        //test ImsR-------------------------------------------------------------
        {
            int IH = 14, IW = 14, OH = 7, OW = 7;
//        int IH = 12, IW = 12, OH = 6, OW = 6;
//        int IH = 16, IW = 16, OH = 8, OW = 8;
            int FH = 3, FW = 3;
            int N = 64, OC = 32;
            int sh = 2, sw = 2, ph = 1, pw = 1;
            for (int ic = 64; ic <= 128; ic++) 
                testCorrect(IH, IW, OH, OW, FH, FW, N, ic, OC, sh, sw, ph, pw);
        }
        
        
        {
            int IH = 22, IW = 22, OH = 7, OW = 7;
            int FH = 3, FW = 3;
            int N = 64, OC = 32;
            int sh = 3, sw = 3, ph = 1, pw = 1;
            for (int ic = 64; ic <= 128; ic++) 
                testCorrect(IH, IW, OH, OW, FH, FW, N, ic, OC, sh, sw, ph, pw);
        }
        
        {
            int IH = 21, IW = 21, OH = 6, OW = 6;
            int FH = 3, FW = 3;
            int N = 64, OC = 32;
            int sh = 3, sw = 3, ph = 1, pw = 1;
            for (int ic = 64; ic <= 128; ic++) 
                testCorrect(IH, IW, OH, OW, FH, FW, N, ic, OC, sh, sw, ph, pw);
        }
    }
}
