package test.cuda.matMul;


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
public class Cuda_matMul_test {
    static { alpha.home("C:\\Users\\Gilgamesh\\Desktop\\Dragon-alpha-v1.2");}
    static Engine eg = alpha.engine.cuda_float32(0, alpha.engine.memp1());
    static {
        CudaFloat32EngineBase base = (CudaFloat32EngineBase) eg.engineBase();
        base.matMul_tf32(true);
    }
     
    static void multiply(float[][] A, float[][] B, float[][] C) {
        //step A = K
        //step B,C = M
        int N=A.length, M=B[0].length, K=B.length;
        for(int k=0;k<K;k++)
            for(int i=0;i<N;i++)
                for(int j=0;j<M;j++)
                    C[i][j]+=A[i][k]*B[k][j];
    }
    //A(K*N), B(K*M), C(N*M), A^T(N*K)
    //C = A^T * B
    static void multiplyT1(float[][] A, float[][] B, float[][] C) {
        //step A = N
        int N=A[0].length, M=B[0].length, K=B.length;
        for(int k=0;k<K;k++)
            for(int i=0;i<N;i++)
                for(int j=0;j<M;j++)
                    C[i][j]+=A[k][i]*B[k][j];
    }
    
    public static void testCorrect(int N, int M, int K) {
        System.out.println("TestCorrect(N, M, K): "+ N + ", " + M + ", " + K);
        float[] A = Vector.random_float_vector(N*K); 
        float[] B = Vector.random_float_vector(K*M); 
        
        Tensor tA = eg.tensor(A, N, K);
        Tensor tB = eg.tensor(B, K, M);
        
        //CPU-------------------------------------------------------------------
        float[][] mA = Matrix.toMatrix(A, K);
        float[][] mB = Matrix.toMatrix(B, M);
        float[][] mC = new float[N][M];
       
        multiply(mA, mB, mC);
        
        float[] C1 = Matrix.toVector(mC, N*M);
        
        //GPU-------------------------------------------------------------------
        Tensor tC1 = eg.matMul(tA, tB).c();
        Tensor tC2 = eg.matMul(eg.empty(N, M).c(), tA, tB).c();
        
        float[] C2 = eg.valueOf(tC1);
        float[] C3 = eg.valueOf(tC2);
        
        //compare---------------------------------------------------------------
        float sp1 = Vector.samePercent_relative(C1, C2);
        float sp2 = Vector.samePercent_relative(C1, C3);
        float zp1 = Vector.zeroPercent(C1);
        float zp2 = Vector.zeroPercent(C2);
        float zp3 = Vector.zeroPercent(C3);
        
        System.out.print("CPU : "); Vector.println(C1, 0, 10);
        System.out.print("GPU1: "); Vector.println(C2, 0, 10);
        System.out.print("GPU2: "); Vector.println(C3, 0, 10);
        
        System.out.println("sp1 = " + sp1);
        System.out.println("sp2 = " + sp2);
        
        System.out.println("zp0 = " + zp1);
        System.out.println("zp1 = " + zp2);
        System.out.println("zp2 = " + zp3);
        
        if(sp1 < 0.999f || sp2 < 0.999f) { throw new RuntimeException(N + " "+ M + " "+K); }
        System.gc();
    }
    
    public static void testSpeed(int N, int M, int K) {
        System.out.format("N, M, K = (%d, %d, %d)\n", N, M, K);
        
        eg.check(false).sync(false);
        float[] A=Vector.random_float_vector(N*K);
        float[] B=Vector.random_float_vector(K*M);
        
        Tensor tA = eg.tensor(A, N, K).c();
        Tensor tB = eg.tensor(B, K, M).c();
        Tensor tC = eg.empty(N, M).c();
       
        int nIter = 1000;
        SimpleTimer timer = new SimpleTimer().record();
        for(int i=0; i<nIter; i++) {
            tC = eg.matMul(tC, tA, tB).c();
        }
        Cuda.deviceSynchronize();
        long dif = timer.record().timeStamp_dif_millis();
        
        System.out.println("total time = " + dif);
        float time = (float) (1.0*dif / nIter);
	double sizeV = 1.0 * N * M * K;
	float performance = (float) ((2 * sizeV * 1.0e-9f) / (time / 1000.0f));
        System.out.format("Size = %f, Time = %f msec, Performance = %f GFlop/s\n", 
                (float)(sizeV/(1024*1024*1024)), time, performance);
    }
    
    public static void main(String[] args)
    {
//          test3();
//        //20  52  2
//        int N = 1024, M = 1148, K = 1024;
//        testCorrect(N, M, K);
//        testSpeed(N, M, K);
//        System.out.println(CudaException.lastException());
        
        Vector.PRINT_DIFFERENT = true;

        for(int n=1; n<=255;n++)
            for(int m=1; m<=255; m++)  
                for(int k=128; k<=132; k++) testCorrect(n, m, k);

        for(int n=1; n<=255; n++)
            for(int m=1; m<=255; m++)  
                for(int k=512; k<=517; k++) testCorrect(n, m, k);
//        
//        int N = 2048, M = 2048, K = 2048;
        int N = 1024, M = 1024, K = 1024;
//        int N = 4096, M = 4096, K = 4096;

//        int N = 256, M = 4096, K = 8192;
//          int N = 256, M = 2048, K = 4096;
//          int N = 256, M = 1024, K = 2048;
//        int N = 256, M = 4096, K = 4096; //5137.520508 GFlop/,s, 0.209000
//        int N = 2000, M = 2000, K = 2048;// 8623.157227 GFlop/s,  8701.008789 GFlop/s
        
//        int N = 256, M = 1000, K = 4096;//6168.093750 GFlop/s,  5874.375488 GFlop/s
//        int N = 256, M = 1000, K = 2048;// 4788.018066 GFlop/s, 4877.097168 GFlop/s
        
        testCorrect(N, M, K);
        testSpeed(N, M, K);
//        testSpeed(N, M, K);
//        testSpeed(N, M, K);
//        testSpeed(N, M, K);
    }
}
