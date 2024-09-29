package test.cuda.dwconv3D;


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
public class Cuda_dwconv3D_test  {
    static { alpha.home("C:\\Users\\Gilgamesh\\Desktop\\Dragon-alpha-v1.2");}
    static Mempool memp = alpha.engine.memp1(alpha.MEM_1GB * 8);
    static Engine eg = alpha.engine.cuda_float32(0, memp, alpha.MEM_1MB * 2048);
    static {
        CudaFloat32EngineBase cu32 = (CudaFloat32EngineBase) eg.engineBase();
    }
     
     public static void testCorrect(
            int IH, int IW, 
            int FH, int FW, 
            int N, int IC, int OC,
            int sh, int sw, int ph, int pw)
    {
        int[] ODim = Cuda_conv3D.output_feature_dim(IH, IW, FH, FW, N, OC, sh, sw, ph, pw);
        int OH = ODim[1], OW = ODim[2];
        
        System.out.println("Test correct:");
        System.out.format("\tXsize( N, IC, IH, IW) = (%d, %d, %d, %d)\n", N, IC, IH, IW);
        System.out.format("\tWsize(OC, IC, FH, FW) = (%d, %d, %d, %d)\n", OC, IC, FH, FW);
        System.out.format("\tYsize( N, OC, OH, OW) = (%d, %d, %d, %d)\n", N, OC, OH, OW);
        
        int[] Gsize = Cuda_conv3D.img2col_matrix_dim(OH, OW, FH, FW, N, IC, OC);
        int GN = Gsize[0], GM = Gsize[1], GK = Gsize[2];
        System.out.format("(GN, GM, GK) = (%d, %d, %d)\n", GN, GM, GK);
        
        int sizeX = N*IC*IH*IW;
	int sizeW = OC*IC*FH*FW;
	int sizeY = N*OC*OH*OW;
        
        float[] X = Vector.random_float_vector(sizeX);
        float[] W = Vector.random_float_vector(sizeW);
        
        Tensor tX = eg.tensor(X, N, IH, IW, IC);
        Tensor tW = eg.tensor(W, OC, FH, FW, IC);
        
        //CPU-------------------------------------------------------------------
        float[][][][] cX = Tense.vectorToTensor_4D(X, N, IH, IW, IC);
        float[][][][] cW = Tense.vectorToTensor_4D(W, OC, FH, FW, IC);
        float[][][][] cY = new float[N][OH][OW][OC];
        
//        Tense.conv3D_naive(
//                cX, IH, IW,
//                cW, FH, FW,
//                cY, OH, OW,
//                N, IC, OC, 
//                sh, sw, ph, pw);
        
        float[] Y1 = Tense.tensor_4DToVector(cY, sizeY);
        
        //GPU-------------------------------------------------------------------
        Tensor tY1 = eg.conv3D(tX, tW, sh, sw, ph, pw);
        Tensor tY2 = eg.conv3D(eg.empty(N, OH, OW, OC).c(), tX, tW, sh, sw);

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
//        if(sp1 <0.999f) throw new RuntimeException();        
        if(sp1 <0.999f || sp2 < 0.999f) throw new RuntimeException();
        System.gc();
    }
     
    public static void testSpeed(int nIter,
            int IH, int IW, 
            int FH, int FW, 
            int OH, int OW,
            int N, int IC, int OC,
            int sh, int sw, int ph, int pw)
    {
        if (OH == - 1) OH = (IH + (ph << 1) - FH) / sh + 1;
        if (OW == - 1) OW = (IW + (pw << 1) - FW) / sw + 1;
        
        System.out.println("Test speed:");
        System.out.format("\tXsize(N, IH, IW, IC) = (%d, %d, %d, %d)\n", N, IH, IW, IC);
        System.out.format("\tWsize(FH, FW, OC) = (%d, %d, %d)\n", FH, FW, OC);
        System.out.format("\tYsize(N, OH, OW, OC) = (%d, %d, %d, %d)\n", N, OH, OW, OC);
        
        int GN = OC;
        int GM = N * OH * OW;
        int GK = FH * FW;
        System.out.format("(GN, GM, GK) = (%d, %d, %d)\n", GN, GM, GK);
        
        int sizeX = N * IC * IH * IW;//256*256
        int sizeW = FH * FW * OC;
        
        float[] X = Vector.random_float_vector(sizeX);
        float[] W = Vector.random_float_vector(sizeW);
        
        Tensor tX = eg.tensor(X, N, IH, IW, IC).c();
        Tensor tW = eg.tensor(W, FH, FW, OC).c();
        Tensor tY = eg.empty(N, OH, OW, OC).c();
        
        //tY = eg.depthwise_conv3D(tY, tX, tW, sh, sw).c();
        tY = eg.depthwise_conv3D(tX, tW, OH, OW, sh, sw, ph, pw).c();
        System.out.println(tY.zero_percent()); 
        
        SimpleTimer timer = new SimpleTimer().record();
        for(int i=0; i<nIter; i++) {
            //tY = eg.depthwise_conv3D(tY, tX, tW, sh, sw).c();
            tY = eg.depthwise_conv3D(tX, tW, OH, OW, sh, sw, ph, pw).c(); eg.delete(tY);
        }
        Cuda.deviceSynchronize();
        
        long div = timer.record().timeStamp_dif_millis();
        float time = 1.0f * div / nIter;
        double sizeV = 1.0 * GN / 1024 * GM / 1024 * GK / 1024;
        float performance = (float) ((sizeV * 1024 * 1024 * 1e-9 * 2 * 1024)/(time * 1e-3));
        
        System.out.format("Size = %f, Time = %f msec, Performance = %f GFlop/s\n", 
                (float)sizeV, time, performance);
    }
    
    public static void main(String[] args) {
//        int FH = 2, FW = 2, ph = 1, pw = 1, sh = 1, sw = 1;
//	int FH = 3, FW = 3, ph = 1, pw = 1, sh = 1, sw = 1;
//	int FH = 4, FW = 4, ph = 2, pw = 2, sh = 1, sw = 1; 
//	int FH = 5, FW = 5, ph = 2, pw = 2, sh = 1, sw = 1;
//	int FH = 6, FW = 6, ph = 3, pw = 3, sh = 1, sw = 1;
//	int FH = 7, FW = 7, ph = 3, pw = 3, sh = 1, sw = 1;
//	int FH = 8, FW = 8, ph = 4, pw = 4, sh = 1, sw = 1;
//	int FH = 9, FW = 9, ph = 4, pw = 4, sh = 1, sw = 1;
//	int FH = 11, FW = 11, ph = 5, pw = 5, sh = 1, sw = 1;
//	int FH = 12, FW = 12, ph = 5, pw = 5, sh = 1, sw = 1;
//	int FH = 13, FW = 13, ph = 6, pw = 6, sh = 1, sw = 1;
//	int FH = 14, FW = 14, ph = 7, pw = 7, sh = 1, sw = 1;
//	int FH = 17, FW = 17, ph = 8, pw = 8, sh = 1, sw = 1;
	int FH = 31, FW = 31, ph = 15, pw = 15, sh = 1, sw = 1;
        
        //int IH = 128, IW = 128, OH = 128, OW = 128, N = 64, IC = 64, OC = 64;
	int IH = 64, IW = 64, OH = 64, OW = 64, N = 128, IC = 128, OC = 128;
	//int IH = 32, IW = 32, OH = 32, OW = 32, N = 128, IC = 256, OC = 256;
	//int IH = 16, IW = 16, OH = 16, OW = 16, N = 64, IC = 512, OC = 512;
        
        testSpeed(500, IH, IW, FH, FW, OH, OW, N, IC, OC, sh, sw, ph, pw);
    }
}
