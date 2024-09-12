package test.cuda.matMul;

import static test.cuda.matMul.Cuda_matMul_test_T1.eg;
import static z.dragon.alpha.Alpha.alpha;
import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.dragon.engine.cuda.CudaFloat32EngineBase;
import z.dragon.engine.cuda.impl.Cuda;
import z.util.lang.SimpleTimer;
import z.util.math.vector.Matrix;
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
public class Cuda_matMul_test_T2  {
    static { alpha.home("C:\\Users\\Gilgamesh\\Desktop\\Dragon-alpha-v1.2");}
    static Engine eg = alpha.engine.cuda_float32(0, alpha.engine.memp1());
    static {
        CudaFloat32EngineBase base = (CudaFloat32EngineBase) eg.engineBase();
        base.matMulT2_tf32(true);
    }
     
    static void multiplyT2(float[][] A, float[][] B, float[][] C) {
        //step A = K
        //step B,C = M
        int N = A.length, M = B.length, K = B[0].length;
        for(int k = 0; k < K; k++)
            for(int i = 0; i < N;i++)
                for(int j = 0; j < M;j++)
                    C[i][j] += A[i][k] * B[j][k];
    }
   
    
    public static void testCorrect(int N, int M, int K) {
        System.out.println("\nTestCorrect(N, M, K): " + N + ", " + M + ", " + K);
        float[] A = Vector.random_float_vector(N*K);
        float[] B = Vector.random_float_vector(M*K);
        
        //CPU-------------------------------------------------------------------
        float[][] mA = Matrix.toMatrix(A, K);
        float[][] mB = Matrix.toMatrix(B, K);
        float[][] mC = new float[N][M];
        
        multiplyT2(mA, mB, mC);
        
        float[] C1 = Matrix.toVector(mC);
        
        //GPU-------------------------------------------------------------------
        Tensor tA = eg.tensor(A, N, K);
        Tensor tB = eg.tensor(B, M, K);
        Tensor tC1 = eg.matMulT2(tA, tB).c();
        Tensor tC2 = eg.matMulT2(eg.empty(N, M).c(), tA, tB).c();
        
        float[] C2 = eg.valueOf(tC1);
        float[] C3 = eg.valueOf(tC2);
        
        //compare---------------------------------------------------------------
        float sp1 = Vector.samePercent_relative(C2, C1);
        float sp2 = Vector.samePercent_relative(C3, C1);
        
        System.out.print("CPU :"); Vector.println(C1, 0, 10);
        System.out.print("GPU1:"); Vector.println(C2, 0, 10);
        System.out.print("GPU2:"); Vector.println(C3, 0, 10);
        System.out.println("sp1 = " + sp1);
        System.out.println("sp2 = " + sp2);
        
        if(sp1 < 0.999f || sp2 < 0.999f) {throw new RuntimeException(N + ", " + M + ", " + K);}
    }
    
    public static void testSpeed(int N, int M, int K) {
        eg.check(false).sync(false);
        float[] A = Vector.random_float_vector(N*K);
        float[] B = Vector.random_float_vector(K*M);
        
        Tensor tA = eg.tensor(A, N, K).c();
        Tensor tB = eg.tensor(B, M, K).c();
        Tensor tC = eg.empty(N, M).c();
       
        int nIter = 1000;
        SimpleTimer timer = new SimpleTimer().record();
        for(int i=0; i<nIter; i++) {
            tC = eg.matMulT2(tC, tA, tB).c();
        }
        Cuda.deviceSynchronize();
        
        long dif =  timer.record().timeStamp_dif_millis();
         
        System.out.println("total time = " + dif);
        float time = (float) (1.0*dif / nIter);
	double sizeV = 1.0 * N * M * K;
	float performance = (float) ((2 * sizeV * 1.0e-9f) / (time / 1000.0f));
        System.out.format("Size = %f, Time = %f msec, Performance = %f GFlop/s\n", 
                (float)(sizeV/(1024*1024*1024)), time, performance);
        
    }
    
    public static void main(String[] args) {
//          test3();
//        //20  52  2
//        int N = 1149, M = 1149, K = 1024;
//        testCorrect(N, M, K);
//        testSpeed(N, M, K);
//        System.out.println(CudaException.lastException());
//        
//        testCorrect(65, 121, 518); return;
        
//        for(int n=1; n<=255;n++)
//            for(int m=1; m<=255; m++)  
//                for(int k=128; k<=132; k++) testCorrect(n, m, k);
//
//        for(int n=1; n<=255; n++)
//            for(int m=1; m<=255; m++)  
//                for(int k=512; k<=517; k++) testCorrect(n, m, k);
        
//        int N = 2048, M = 2048, K = 2048;
        int N = 4096, M = 4096, K = 4096;
//        int N = 1024, M = 1024, K = 1024;
//        int N = 256, M = 4096, K = 8192;
//          int N = 256, M = 2048, K = 4096;
//          int N = 256, M = 1024, K = 2048;
//        int N = 256, M = 4096, K = 4096; //5137.520508 GFlop/,s, 0.209000
//        int N = 2000, M = 2000, K = 2048;// 8623.157227 GFlop/s,  8701.008789 GFlop/s
        
//        int N = 256, M = 1000, K = 4096;//6168.093750 GFlop/s,  5874.375488 GFlop/s
//        int N = 256, M = 1000, K = 2048;// 4788.018066 GFlop/s, 4877.097168 GFlop/s
        
//        testCorrect(N/2, M/2, K/2);
        testSpeed(N, M, K);
//        testSpeed(256, 1024, 4096);
    }
}
