package test.cuda.matMul;


import static test.cuda.matMul.Cuda_matMul_test.eg;
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
public class Cuda_matMul_test_T1 {
    static { alpha.home("C:\\Users\\Gilgamesh\\Desktop\\Dragon-alpha-v1.2");}
    static Engine eg = alpha.engine.cuda_float32(0, alpha.engine.memp1());
    static {
        CudaFloat32EngineBase base = (CudaFloat32EngineBase) eg.engineBase();
        base.matMulT1_tf32(true);
    }
    
    //A(K*N), B(K*M), C(N*M), A^T(N*K)
    //C = A^T * B
    static void multiplyT1(float[][] A, float[][] B, float[][] C) {
        //step A = N
        int N=A[0].length, M=B[0].length, K=B.length;
        for(int k=0;k<K;k++)
            for(int i=0;i<N;i++)
                for(int j=0;j<M;j++)
                    C[i][j] += A[k][i]*B[k][j];
    }
    
    public static void testCorrect(int N, int M, int K) {
        eg.sync(false);
        System.out.println("TestCorrect:" + N +",  " + M + ",  " + K);
        float[] A = Vector.random_float_vector(K * N);
        float[] B = Vector.random_float_vector(K * M);
        
        //CPU-------------------------------------------------------------------
        float[][] mA = Matrix.toMatrix(A, N);
        float[][] mB = Matrix.toMatrix(B, M);
        float[][] mC = new float[N][M];
        
        multiplyT1(mA, mB, mC);
        
        float[] C1 = Matrix.toVector(mC, N*M);
        
        //GPU-------------------------------------------------------------------
        Tensor tA = eg.tensor(A, K, N).c();
        Tensor tB = eg.tensor(B, K, M).c();

        Tensor tC1 = eg.matMulT1(tA, tB).c();
        Tensor tC2 = eg.matMulT1(eg.empty(N, M).c(), tA, tB).c();
        
        float[] vA = tA.value();
        float[] vB = tB.value();
        float spA = Vector.samePercent_absolute(vA, A); System.out.println("spA = " + spA);
        float spB = Vector.samePercent_absolute(vB, B); System.out.println("spB = " + spB);
        
        float[] C2 = eg.valueOf(tC1);
        float[] C3 = eg.valueOf(tC2);
        
        //compare---------------------------------------------------------------
        float sp1 = Vector.samePercent_relative(C2, C1, 1e-3f);
        float sp2 = Vector.samePercent_relative(C3, C1, 1e-3f);
        
        System.out.print("CPU :"); Vector.println(C1, 0, 10);
        System.out.print("GPU1:" ); Vector.println(C2, 0, 10);
        System.out.print("GPU2:" ); Vector.println(C3, 0, 10);
        System.out.println("sp1 = " + sp1);
        System.out.println("sp2 = " + sp2);
        
        if(sp1 < 0.999f || sp2 < 0.999f) { throw new RuntimeException(N+" "+M+" "+K); }
    }
    
    public static void testSpeed(int N, int M, int K) {
        eg.check(false).sync(false);

        float[] A = Vector.random_float_vector(K*N);
        float[] B = Vector.random_float_vector(K*M);
        
        Tensor tA = eg.tensor(A, K, N).c();
        Tensor tB = eg.tensor(B, K, M).c();
        Tensor tC = eg.empty(N, M).c();
       
        int nIter = 1000;
        SimpleTimer timer = new SimpleTimer().record();
        for(int i=0; i<nIter; i++) {
            tC = eg.matMulT1(tC, tA, tB).c();
        }
        Cuda.deviceSynchronize();
        long dif = timer.record().timeStamp_dif_millis();
        
        System.out.println(eg);
        System.out.println("total time = " + dif);
        
        float msecPerMatrixMul = (float) (1.0*dif / nIter);
	double flopsPerMatrixMul = 2.0 * N * M * K;
	float gigaFlops = (float) ((flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f));
        System.out.print(msecPerMatrixMul + " msec, ");
        System.out.println(gigaFlops+" GFlop/s");
    }
    
    public static void main(String[] args)
    {
//        Vector.PRINT_DIFFERENT = true;
        try {
//        int N = 380, M = 380, K = 1024*16;
//        testCorrect(N, M, K);
//        testSpeed(N, M, K);
//        System.out.println(CudaException.lastException());
//        
//            for(int N=1; N<=64;N++)
//                for(int M=4; M<=64; M++)  
//                    for(int K=2; K<=32; K++) testCorrect(N, M, K);
            
//            testCorrect(9, 9, 512);
            
            int N = 4096, M = 4096, K = 4096;
            
            testCorrect(1024, 1024, 1024);
            testSpeed(2048, 2048, 2048);
            testSpeed(N, M, K);

//            for(int n=1; n<=255;n++)
//                for(int m=1; m<=255; m++)  
//                    for(int k=128; k<=132; k++) testCorrect(n, m, k);
//
//            for(int n=1; n<=255; n++)
//                for(int m=1; m<=255; m++)  
//                    for(int k=512; k<=517; k++) testCorrect(n, m, k);
            
        }
        catch(Exception e) {
            e.printStackTrace();
        }
    }
}
