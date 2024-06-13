/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.util.math.vector;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadFactory;
import z.util.lang.SimpleTimer;
import static z.util.math.vector.MatMul_naive.matMul1;

/**
 *
 * @author Gilgamesh
 */
public class MatMul 
{
    static final ThreadFactory daemonThreadFactory = (Runnable r) -> { Thread t = new Thread(r); t.setDaemon(true); return t; };
    static final ExecutorService exec = Executors.newFixedThreadPool(16, daemonThreadFactory); 
    
    //v = 8*8 = 64
    public static float[][] As = new float[4][32];//4*8
    public static float[][] Bs = new float[4][32];//4*8
    
    public static void outer_product(float[] v, int ty, int tx) {
        float a[] = new float[8];
        float b[] = new float[8];
        
        a[0] = As[tx][ty*8 + 0];
        a[1] = As[tx][ty*8 + 1];
        a[2] = As[tx][ty*8 + 2];
        a[3] = As[tx][ty*8 + 3];
        a[4] = As[tx][ty*8 + 4];
        a[5] = As[tx][ty*8 + 5];
        a[6] = As[tx][ty*8 + 6];
        a[7] = As[tx][ty*8 + 7];
        
        b[0] = Bs[tx][ty*8 + 0];
        b[1] = Bs[tx][ty*8 + 1];
        b[2] = Bs[tx][ty*8 + 2];
        b[3] = Bs[tx][ty*8 + 3];
        b[4] = Bs[tx][ty*8 + 4];
        b[5] = Bs[tx][ty*8 + 5];
        b[6] = Bs[tx][ty*8 + 6];
        b[7] = Bs[tx][ty*8 + 7];
              
        v[0*8 + 0] += a[0]*b[0]; v[0*8 + 1] += a[0]*b[1]; v[0*8 + 2] += a[0]*b[2]; v[0*8 + 3] += a[0]*b[3];
        v[1*8 + 0] += a[1]*b[0]; v[1*8 + 1] += a[1]*b[1]; v[1*8 + 2] += a[1]*b[2]; v[1*8 + 3] += a[1]*b[3];
        v[2*8 + 0] += a[2]*b[0]; v[2*8 + 1] += a[2]*b[1]; v[2*8 + 2] += a[2]*b[2]; v[2*8 + 3] += a[2]*b[3];
        v[3*8 + 0] += a[3]*b[0]; v[3*8 + 1] += a[3]*b[1]; v[3*8 + 2] += a[3]*b[2]; v[3*8 + 3] += a[3]*b[3];
        v[4*8 + 0] += a[4]*b[0]; v[4*8 + 1] += a[4]*b[1]; v[4*8 + 2] += a[4]*b[2]; v[4*8 + 3] += a[4]*b[3];
        v[5*8 + 0] += a[5]*b[0]; v[5*8 + 1] += a[5]*b[1]; v[5*8 + 2] += a[5]*b[2]; v[5*8 + 3] += a[5]*b[3];
        v[6*8 + 0] += a[6]*b[0]; v[6*8 + 1] += a[6]*b[1]; v[6*8 + 2] += a[6]*b[2]; v[6*8 + 3] += a[6]*b[3];
        v[7*8 + 0] += a[7]*b[0]; v[7*8 + 1] += a[7]*b[1]; v[7*8 + 2] += a[7]*b[2]; v[7*8 + 3] += a[7]*b[3];

        v[0*8 + 4] += a[0]*b[4]; v[0*8 + 5] += a[0]*b[5]; v[0*8 + 6] += a[0]*b[6]; v[0*8 + 7] += a[0]*b[7];
        v[1*8 + 4] += a[1]*b[4]; v[1*8 + 5] += a[1]*b[5]; v[1*8 + 6] += a[1]*b[6]; v[1*8 + 7] += a[1]*b[7];
        v[2*8 + 4] += a[2]*b[4]; v[2*8 + 5] += a[2]*b[5]; v[2*8 + 6] += a[2]*b[6]; v[2*8 + 7] += a[2]*b[7];
        v[3*8 + 4] += a[3]*b[4]; v[3*8 + 5] += a[3]*b[5]; v[3*8 + 6] += a[3]*b[6]; v[3*8 + 7] += a[3]*b[7];
        v[4*8 + 4] += a[4]*b[4]; v[4*8 + 5] += a[4]*b[5]; v[4*8 + 6] += a[4]*b[6]; v[4*8 + 7] += a[4]*b[7];
        v[5*8 + 4] += a[5]*b[4]; v[5*8 + 5] += a[5]*b[5]; v[5*8 + 6] += a[5]*b[6]; v[5*8 + 7] += a[5]*b[7];
        v[6*8 + 4] += a[6]*b[4]; v[6*8 + 5] += a[6]*b[5]; v[6*8 + 6] += a[6]*b[6]; v[6*8 + 7] += a[6]*b[7];
        v[7*8 + 4] += a[7]*b[4]; v[7*8 + 5] += a[7]*b[5]; v[7*8 + 6] += a[7]*b[6]; v[7*8 + 7] += a[7]*b[7];
    }
    
    public static void load(
            float[] A,//[N, K]
            float[] B,//[K, M]
            int M, int K,
            int ty, int tx, int n, int m, int k) 
    {
        int Aoffset = (ty*8 + n)*K + tx;//A[n + ty*8, k: tx]
        int Boffset = tx*M + (ty*8 + m);//B[k: tx, n + ty*8]
        
        float[] av = new float[8];
        float[] bv = new float[8];
        
        for(int i=0; i<8; i++) av[i] = A[Aoffset + K*i];
        for(int i=0; i<8; i++) bv[i] = B[Boffset + i];
        
        for(int i=0; i<8; i++) As[tx][ty] = av[i];
        for(int i=0; i<8; i++) Bs[tx][ty] = bv[i];
    }
    
    public static void matMul(float[] A, float[] B, float[] C, int N, int M, int K) throws Exception {
        Future[] fts = new Future[16];
        
        for(int m=0; m<M; m+=32)
        for(int n=0; n<N; n+=32) {
            float[][] v = new float[16][64]; 
            for(int k=0; k<K; k+=4) {
                //=======[load]=================================================
                for(int y=0; y<4; y++)
                for(int x=0; x<4; x++) {
                    int ty = y, tx = x, tn = n, tm = m, tk = k;
                    fts[y*4 + x] = exec.submit(()->{ load(A, B, M, K, ty, tx, tn, tm, tk); });
                }
                for(Future ft : fts) ft.get();
//                
//                //=======[outer-product]========================================
                for(int y=0; y<4; y++)
                for(int x=0; x<4; x++) {
                    int ty = y, tx = x;
//                    fts[y*4 + x] = exec.submit(()->{ outer_product(v[ty*4 + tx], ty, tx); });
                }
//                for(Future ft : fts) ft.get();
            }
        }
    }
    
    public static void main(String[] args) throws Exception
    {
       int N = 1024, M = 1024, K = 1024;
        
        float[] A = Vector.random_float_vector(N*K);
        float[] B = Vector.random_float_vector(K*M);
        float[] C = new float[N*M];
//        matMul1(A, B, C, N, M, K);
        
        int nIter = 10;
        SimpleTimer timer = SimpleTimer.clock();
        for(int i=0; i<nIter; i++) {
            matMul(A, B, C, N, M, K);
        }
        exec.shutdown();
        
        long dif = timer.record().timeStamp_dif_millis();
        
        System.out.println("total time = " + dif);
        float time = (float) (1.0*dif / nIter);
	double sizeV = 1.0 * N * M * K;
	float performance = (float) ((2 * sizeV * 1.0e-9f) / (time / 1000.0f));
        System.out.format("Size = %f, Time = %f msec, Performance = %f GFlop/s\n", 
                (float)(sizeV/(1024*1024*1024)), time, performance);
        
        
    }
}
