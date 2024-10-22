/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package test.cuda.matMul;

import static z.dragon.alpha.Alpha.alpha;
import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.dragon.engine.cuda.CudaFloat32EngineBase;
import z.dragon.engine.cuda.impl.Cuda;
import z.util.lang.SimpleTimer;
import z.util.math.vector.Vector;

/**
 *
 * @author Gilgamesh
 */
public class Cuda_batchMatMulT2_test {
    static { alpha.home("C:\\Users\\Gilgamesh\\Desktop\\Dragon-alpha-v1.2");}
    static Engine eg = alpha.engine.cuda_float32(0, alpha.engine.memp1());
    static {
        CudaFloat32EngineBase cu32 = (CudaFloat32EngineBase) eg.engineBase();
//        cu32.batchMatMulT2_useTexture(false);
         cu32.tf32(true);
    }
    
    static void matMulT2(float[][] A, float[][] B, float[][] C) {
        int N = A.length, M = B.length, K = B[0].length;
        for(int k = 0;  k <K; k++)
            for(int i = 0; i < N; i++)
                for(int j = 0; j < M; j++)
                    C[i][j] += A[i][k] * B[j][k];
    }
    
    static void bmmT2(float[][][][] A, float[][][][] B, float[][][][] C)  {
        int dim0 = A.length;
        int dim1 = A[0].length;
        
        for(int d0=0; d0<dim0; d0++) 
            for(int d1=0; d1<dim1; d1++)
                matMulT2(A[d0][d1], B[d0][d1], C[d0][d1]);
    }
    
    public static void testCorrect(int Batch0, int Batch1, int N, int M, int K) {
        System.out.println("TestCorrect:");
        System.out.format("[Batch0, Batch1, N, M, K] = [%d, %d, %d, %d, %d]\n",
                Batch0, Batch1, N, M, K);
        
        int sizeA = Batch0 * Batch1 * N * K;
        int sizeB = Batch0 * Batch1 * M * K;
        float[] A = Vector.random_float_vector(sizeA);
        float[] B = Vector.random_float_vector(sizeB);
        Tensor tA = eg.tensor(A, Batch0 * Batch1, N, K).c();
        Tensor tB = eg.tensor(B, Batch0 * Batch1, M, K).c();
        
        //CPU-------------------------------------------------------------------
        float[][][][] mA = Vector.to4D(A, Batch0, Batch1, N, K);
        float[][][][] mB = Vector.to4D(B, Batch0, Batch1, M, K);
        float[][][][] mC = new float[Batch0][Batch1][N][M];
        
        bmmT2(mA, mB, mC);
        
        float[] C1 = Vector.flatten(mC);
        System.out.print("CPU = "); Vector.println(C1, 0, 10);
        
        //GPU-------------------------------------------------------------------
        Tensor tC = eg.empty(Batch0, Batch1, N, M);
        tC = eg.batchMatMulT2(tC, tA, tB).c();
        
        float[] C2 = tC.value();
        System.out.print("GPU = "); Vector.println(C2, 0, 10);
        
        //compare---------------------------------------------------------------
        float sp = Vector.samePercent_relative(C2, C1);
        System.out.println("sp = " + sp);
        
        eg.delete(tA, tB, tC);
        if(sp != 1.0f) {throw new RuntimeException(N+" "+M+" "+K);}
    }
    
    public static void testSpeed(int Batch0, int Batch1, int N, int M, int K) {
        eg.check(false);
        eg.sync(false);
        
        System.out.println("TestCorrect:");
        System.out.format("[Batch0, Batch1, N, M, K] = [%d, %d, %d, %d, %d]\n",
                Batch0, Batch1, N, M, K);
        
        int sizeA = Batch0 * Batch1 * N * K;
        int sizeB = Batch0 * Batch1 * K * M;
        float[] A = Vector.random_float_vector(sizeA);
        float[] B = Vector.random_float_vector(sizeB);
        Tensor tA = eg.tensor(A, Batch0, Batch1, N, K).c();
        Tensor tB = eg.tensor(B, Batch0, Batch1, M, K).c();
        Tensor tC = eg.empty(Batch0, Batch1, N, M).c();
        
        int nIter = 500;
        SimpleTimer timer = new SimpleTimer().record();
        for(int i=0; i<nIter; i++) {
            tC = eg.batchMatMulT2(tC, tA, tB);
        }
        Cuda.deviceSynchronize();
        
        long dif = timer.record().timeStamp_dif_millis();
        float msecPerMatrixMul = (float) (1.0*dif / nIter);
	double flopsPerMatrixMul = Batch0 * Batch1 * 2.0 * N * M * K;
	float gigaFlops = (float) ((flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f));
        float size = (float) (flopsPerMatrixMul / 2 / 1024 / 1024 / 1024);
        
        System.out.print("size = " + size );
        System.out.print(", Time = " + msecPerMatrixMul + " msec");
        System.out.println(", Performance = "+ gigaFlops+" GFlop/s");
    }
    
    public static void main(String[] args) {
//        Vector.PRINT_DIFFERENT = true;
//        for(int K=7; K<=9; K++)
//        for(int N=1; N <= 82; N++)
//            for(int M=1; M <= 96; M++)  
//                testCorrect(5, 3, N, M, K);
        
        //[64, 256, 128]
//        [64, 8, 256, 32] * [64, 8, 32, 256]
        int Batch0 = 8, Batch1 = 8;
        //int N = 512, M = 512, K = 512;
        
//        int N = 112 * 112, M = 128, K = 128;
//        int N = 56 * 56, M = 256, K = 256;
//        int N = 28 * 28, M = 512, K = 512;
//        int N = 14 * 14, M = 512, K = 512;
        
//        int N = 128, M = 128, K = 112 * 112;
//        int N = 256, M = 256, K = 56 * 56;
        int N = 512, M = 512, K = 28 * 28;
//        int N = 1024, M = 1024, K = 14 * 14;//low
        
//        testCorrect(Batch0, Batch1, N, M, K);
        //testSpeed(Batch0*2, Batch1, N, M, K);
        testSpeed(Batch0, Batch1, N, M, K);
    }
}
