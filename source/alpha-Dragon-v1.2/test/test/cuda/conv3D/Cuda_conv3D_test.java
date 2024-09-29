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
public class Cuda_conv3D_test 
{
    static { alpha.home("C:\\Users\\Gilgamesh\\Desktop\\Dragon-alpha-v1.2");}
    static Mempool memp = alpha.engine.memp1(alpha.MEM_1GB * 8);
    static Engine eg = alpha.engine.cuda_float32(0, memp, alpha.MEM_1MB * 2048);
    static {
        CudaFloat32EngineBase cu32 = (CudaFloat32EngineBase) eg.engineBase();
//        cu32.conv3D_use_texture(false);
//        cu32.conv3D_remode(false);
//        cu32.conv3D_useTexture(false);
    }
     
     public static void testCorrect(
            int IH, int IW, 
            int FH, int FW, 
            int N, int IC, int OC,
            int sh, int sw, int ph, int pw)
    {
//        eg.sync(false);
        
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
     
    public static void testSpeed(
            int IH, int IW, 
            int FH, int FW, 
            int N, int IC, int OC,
            int sh, int sw, int ph, int pw)
    {
        int[] ODim = Cuda_conv3D.output_feature_dim(IH, IW, FH, FW, N, OC, sh, sw, ph, pw);
        int OH = ODim[1], OW = ODim[2];
        
        System.out.println("Test speed:");
        System.out.format("\tXsize( N, IC, IH, IW) = (%d, %d, %d, %d)\n", N, IC, IH, IW);
        System.out.format("\tWsize(OC, IC, FH, FW) = (%d, %d, %d, %d)\n", OC, IC, FH, FW);
        System.out.format("\tYsize( N, OC, OH, OW) = (%d, %d, %d, %d)\n", N, OC, OH, OW);
        
        int[] Gsize = Cuda_conv3D.img2col_matrix_dim(OH, OW, FH, FW, N, IC, OC);
        int GN=Gsize[0], GM=Gsize[1], GK=Gsize[2];
        System.out.format("(GN, GM, GK) = (%d, %d, %d)\n", GN, GM, GK);
        
        int sizeX = N*IC*IH*IW;//256*256
	int sizeW = OC*IC*FH*FW;
        
        float[] X=Vector.random_float_vector(sizeX);
        float[] W=Vector.random_float_vector(sizeW);
        
        Tensor tX = eg.tensor(X, N, IH, IW, IC);
        Tensor tW = eg.tensor(W, OC, FH, FW, IC);
        Tensor tY = eg.empty(N, OH, OW, OC).c();
        
        tY = eg.conv3D(tY, tX, tW, sh, sw).c();//async
        
        int nIter = 1000;  
        eg.check(false).sync(false);
        SimpleTimer timer = new SimpleTimer().record();
        for(int i=0; i<nIter; i++) {
//            tY = eg.conv3D(tX, tW, sh, sw, ph, pw); eg.delete(tY.c());//sync
            tY = eg.conv3D(tY, tX, tW, sh, sw).c();//async
        }
        Cuda.deviceSynchronize();
        
        long div = timer.record().timeStamp_dif_millis();
        float time = 1.0f*div/nIter;
        double sizeV = 1.0 * GN/1024 * GM/1024 * GK/1024;
        float performance = (float) ((sizeV*1024*1024 * 1e-9 * 2 * 1024)/(time * 1e-3));
        
        System.out.format("Size = %f, Time = %f msec, Performance = %f GFlop/s\n", 
                (float)sizeV, time, performance);
    }
    
    public static void main(String[] args) {
//        int FH = 9, FW = 9, sh = 1, sw = 1, ph = 4, pw = 4;
//        int FH = 8, FW = 8, sh = 1, sw = 1, ph = 4, pw = 4;
        int FH = 7, FW = 7, sh = 1, sw = 1, ph = 3, pw = 3;
//        int FH = 1, FW = 7, sh = 1, sw = 1, ph = 0, pw = 3;
//        int FH = 5, FW = 5, sh = 1, sw = 1, ph = 2, pw = 2;
//        int FH = 3, FW = 3, sh = 1, sw = 1, ph = 1, pw = 1;
//        int IH = 128, IW = 128, IC =   3, OC =  64, N = 128;//false
//         int IH = 128, IW = 128, IC =   3, OC =  64, N = 128;//false
        
        int IH = 64, IW = 64, N = 64, IC = 64, OC = 64;
//        int IH = 32, IW = 32,  N = 128, IC = 64, OC = 64;
//        int IH = 16, IW = 16, IC = 256, OC = 256, N = 64;
//        int IH = 16, IW = 16, IC = 128, OC = 128, N = 128;
//        int IH =  8, IW =  8, IC = 128, OC = 256, N = 128;
//        int IH =  8, IW =  8, IC = 256, OC = 256, N = 128;
        
//        testCorrect(IH, IW, FH, FW, N, IC, OC, sh, sw, ph, pw);
        testSpeed(IH, IW, FH, FW, N, IC, OC, sh, sw, ph, pw);
        System.exit(0);
        
//        int FH = 7, FW = 7, ph = 3, pw = 3, sh = 1, sw = 1;
//        int FH = 6, FW = 6, ph = 3, pw = 3, sh = 1, sw = 1;
//        int FH = 5, FW = 5, ph = 2, pw = 2, sh = 1, sw = 1;
//        int FH = 4, FW = 4, ph = 1, pw = 1, sh = 1, sw = 1;
//        int FH = 3, FW = 3, ph = 1, pw = 1, sh = 1, sw = 1;
        
//        int IH = 128, IW = 128, OH = 128, OW = 128, N =  32, IC = 64, OC = 64;
//        int IH =  64, IW =  64, OH =  64, OW =  64, N = 128, IC = 64, OC = 64;
//        int IH =  32, IW =  32, OH =  32, OW =  32, N = 128, IC = 128, OC = 128;
//        int IH =  24, IW = 24, OH =  24, OW =  24, N = 128, IC = 128, OC = 128;
//        int IH =  16, IW =  16, OH =  16, OW =  16, N = 128, IC = 256, OC = 256;
//        int IH =   8, IW =   8, OH =   8, OW =   8, N = 128, IC = 512, OC = 512;
//        int IH =   4, IW =   4, OH =   4, OW =   4, N = 128, IC = 1024, OC = 1024;

//        testSpeed(IH, IW, FH, FW, N, IC*2, OC*2, sh, sw, ph, pw);
        
//        int IH = 2, IW = 2;//(33 - 4 + 2)/1 + 1
//        int FH = 3, FW = 3;
//        int sh = 2, sw = 2, ph = 1, pw = 1;
//        int N = 256;
//        int IC = 512, OC = 512;
//        testCorrect(IH, IW, FH, FW, N, IC, OC, sh, sw, ph, pw);
//            int IC = 16, OC = 128;//9*4=36 
//            for(int oc = 1; oc <= 128; oc++) {
//                testCorrect(IH, IW, FH, FW, N, IC, oc, sh, sw, ph, pw);
//            }
//        }
        
//        testSpeed(128, 128, 3, 3, 256,  4,  64, 1, 1, 1, 1);//error(all 0s): [256, 128, 128,   4] -> [256, 128, 128,  64]
        
//        testCorrect(128, 128, 3, 3, 256,  3,  64, 1, 1, 1, 1);//error(all 0s): [256, 128, 128,   4] -> [256, 128, 128,  64]
//        testCorrect(128, 128, 3, 3, 256,  64,  64, 1, 1, 1, 1);//error(all 0s): [256, 128, 128,  64] -> [256, 128, 128,  64]
        
        
        //Vector.PRINT_DIFFERENT = true;
        
        //test W11==============================================================
//        {
//            int IH = 32, IW = 32;
//            int FH = 1, FW = 1;
//            int N = 4;
//            int IC = 128, OC = 16;
//            int sh = 1, sw = 1, ph = 0, pw = 0;
//            for(int oc = 1; oc <= 128; oc++) 
//                testCorrect(IH, IW, FH, FW, N, IC, oc, sh, sw, ph, pw);
//        }
//       
//        {
//            int IH = 64, IW = 64;
//            int FH = 1, FW = 1;
//            int N = 4;
//            int IC = 128, OC = 255;
//            int sh = 1, sw = 1, ph = 0, pw = 0;
//            testCorrect(IH, IW, FH, FW, N, IC, OC, sh, sw, ph, pw);
//            testSpeed(IH, IW, FH, FW, N*2, IC, OC, sh, sw, ph, pw);
//        }
        
        //test np===============================================================
//        int IH = 33, IW = 33;//(33 - 4 + 2)/1 + 1
//	int FH = 5, FW = 5;//FH = 4, FW = 4
//	int sh = 2, sw = 2, ph = 0, pw = 0;
//	int N = 4;
//	int IC = 16, OC = 128;//9*4=36 
//        for(int oc = 1; oc <= 128; oc++) 
//            testCorrect(IH, IW, FH, FW, N, IC, oc, sh, sw, ph, pw);

//        int IH = 35, IW = 35;//(33 - 4 + 2)/1 + 1
//	int FH = 4, FW = 4;//FH = 4, FW = 4
//	int sh = 1, sw = 1, ph = 0, pw = 0;
//	int N = 4;
//	int IC = 64, OC = 192;//9*4=36 
//        testCorrect(IH, IW, FH, FW, N, IC, OC, sh, sw, ph, pw);
//        testSpeed(IH, IW, FH, FW, N*2, IC, OC, sh, sw, ph, pw);
        
        //test common===========================================================
//        {
//            int IH = 16, IW = 16;//(33 - 4 + 2)/1 + 1
//            int FH = 3, FW = 3;
//            int sh = 2, sw = 2, ph = 1, pw = 1;
//            int N = 4;
//            int IC = 16, OC = 128;//9*4=36 
//            for(int oc = 1; oc <= 128; oc++) {
//                testCorrect(IH, IW, FH, FW, N, IC, oc, sh, sw, ph, pw);
//            }
//        }
//        
//        {
//            int IH = 16, IW = 16;//(33 - 4 + 2)/1 + 1
//            int FH = 4, FW = 4;
//            int sh = 2, sw = 2, ph = 1, pw = 1;
//            int N = 4;
//            int IC = 16, OC = 128;//9*4=36 
//            for(int oc = 1; oc <= 128; oc++) {
//                testCorrect(IH, IW, FH, FW, N, IC, oc, sh, sw, ph, pw);
//            }
//        }
//        
//        {
//            int IH = 16, IW = 16;//(33 - 4 + 2)/1 + 1
//            int FH = 5, FW = 5;
//            int sh = 2, sw = 2, ph = 2, pw = 2;
//            int N = 4;
//            int IC = 16, OC = 128;//9*4=36 
//            for(int oc = 1; oc <= 128; oc++) {
//                testCorrect(IH, IW, FH, FW, N, IC, oc, sh, sw, ph, pw);
//            }
//        }
//        
//        {
//            int IH = 8, IW = 8;//(33 - 4 + 2)/1 + 1
//            int FH = 5, FW = 5;
//            int sh = 2, sw = 2, ph = 2, pw = 2;
//            int N = 64;
//            int IC = 16, OC = 128;//9*4=36 
//            for(int oc = 1; oc <= 128; oc++) {
//                testCorrect(IH, IW, FH, FW, N, IC, oc, sh, sw, ph, pw);
//            }
//            
//            sh = 1; sw = 1;
//            for(int oc = 1; oc <= 128; oc++) {
//                testCorrect(IH, IW, FH, FW, N, IC, oc, sh, sw, ph, pw);
//            }
//        }
//        
//        {
//            int IH = 8, IW = 8;//(33 - 4 + 2)/1 + 1
//            int FH = 6, FW = 6;
//            int sh = 2, sw = 2, ph = 2, pw = 2;
//            int N = 64;
//            int IC = 16, OC = 128;//9*4=36 
//            for(int oc = 1; oc <= 128; oc++) {
//                testCorrect(IH, IW, FH, FW, N, IC, oc, sh, sw, ph, pw);
//            }
//            
//            sh = 1; sw = 1;
//            for(int oc = 1; oc <= 128; oc++) {
//                testCorrect(IH, IW, FH, FW, N, IC, oc, sh, sw, ph, pw);
//            }
//        }
//        
//        {
//            int IH = 8, IW = 8;//(33 - 4 + 2)/1 + 1
//            int FH = 5, FW = 5;
//            int sh = 2, sw = 2, ph = 1, pw = 1;
//            int N = 64;
//            int IC = 16, OC = 128;//9*4=36 
//            
//            testCorrect(IH, IW, FH, FW, N, IC, 61, sh, sw, ph, pw);
//                
//            for(int oc = 1; oc <= 128; oc++) {
//                testCorrect(IH, IW, FH, FW, N, IC, oc, sh, sw, ph, pw);
//            }
//            
//            sh = 1; sw = 1;
//            for(int oc = 1; oc <= 128; oc++) {
//                testCorrect(IH, IW, FH, FW, N, IC, oc, sh, sw, ph, pw);
//            }
//        }
//        
//        {
//            int IH = 8, IW = 8;//(33 - 4 + 2)/1 + 1
//            int FH = 7, FW = 7;
//            int sh = 2, sw = 2, ph = 3, pw = 3;
//            
//            int N = 64;
//            int IC = 16, OC = 128;//9*4=36 
//            
//            testCorrect(IH, IW, FH, FW, N, IC, 61, sh, sw, ph, pw);
//                
//            for(int oc = 1; oc <= 128; oc++) {
//                testCorrect(IH, IW, FH, FW, N, IC, oc, sh, sw, ph, pw);
//            }
//            
//            sh = 1; sw = 1;
//            for(int oc = 1; oc <= 128; oc++) {
//                testCorrect(IH, IW, FH, FW, N, IC, oc, sh, sw, ph, pw);
//            }
//        }
//        
//        int IH = 32, IW = 32;//(33 - 4 + 2)/1 + 1
//	int FH = 4, FW = 4;//FH = 4, FW = 4
//	int sh = 2, sw = 2, ph = 1, pw = 1;
//	int N = 16;
//        int IC = 64, OC = 252;//9*4=36 

//        int IH = 32, IW = 32;//(33 - 4 + 2)/1 + 1
//	int FH = 3, FW = 3;//FH = 4, FW = 4
//	int sh = 2, sw = 2, ph = 1, pw = 1;
//	int N = 32;
//        int IC = 64, OC = 252;//9*4=36 

//        int IH = 32, IW = 32;//(33 - 4 + 2)/1 + 1
//	int FH = 5, FW = 5;//FH = 4, FW = 4
//	int sh = 2, sw = 2, ph = 2, pw = 2;
//	int N = 16;
//        int IC = 64, OC = 248;//9*4=36 

//        int IH = 4, IW = 4;//(33 - 4 + 2)/1 + 1
//	int FH = 5, FW = 5;//FH = 4, FW = 4
//	int sh = 2, sw = 2, ph = 2, pw = 2;
//	int N = 129;
//        int IC = 128, OC = 512;//9*4=36 

        //testCorrect(IH, IW, FH, FW, N, IC, OC, sh, sw, ph, pw);
        //testSpeed(IH, IW, FH, FW, N*2, IC, OC, sh, sw, ph, pw);
    }
}
