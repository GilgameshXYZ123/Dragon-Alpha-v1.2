/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.util.math.vector;

import z.util.lang.SimpleTimer;

/**
 *
 * @author Gilgamesh
 */
public class MatMul_naive 
{
    //Size = 1.000000, Time = 331.700012 msec, Performance = 6.474174 GFlop/s
    static void matMul7(float[] A, float[] B, float[] C, int N, int M, int K) {
        float abuf[] = new float[8 * 8];
        float a[] = new float[8];
        float b[] = new float[8];
        
        for(int n=0; n<N; n += 8)
        for(int m=0; m<M; m += 8) {
            float[] v = new float[64];
            for(int ok=0; ok<K; ok += 8) {
                for(int i=0; i<8; i++)
                for(int ik=0; ik<8; ik++) abuf[ik*8 + i] = A[(n + i)*K + ok + ik]; 
               
                for(int ik=0; ik<8; ik++) {
                    for(int i=0; i<8; i++) a[i] = abuf[ik*8 + i];
                    
                    //System.arraycopy(B, k*M + m, b, 0,  8);
                    int k = ok + ik;
                    b[0] = B[k*M + (m + 0)];
                    b[1] = B[k*M + (m + 1)];
                    b[2] = B[k*M + (m + 2)];
                    b[3] = B[k*M + (m + 3)];
                    b[4] = B[k*M + (m + 4)];
                    b[5] = B[k*M + (m + 5)];
                    b[6] = B[k*M + (m + 6)];
                    b[7] = B[k*M + (m + 7)];
               
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
            }
            
            for(int i=0; i<8; i++)
            for(int j=0; j<8; j++)
                C[(n + i)*M + (m + j)] = (float) v[i*8 + j];
        }
    }
    
    //Size = 1.000000, Time = 375.399994 msec, Performance = 5.720521 GFlop/s
    static void matMul6(float[] A, float[] B, float[] C, int N, int M, int K) {
        float abuf[] = new float[8 * 8];
        
        float a[] = new float[8];
        float b[] = new float[8];
        
        for(int n=0; n<N; n += 8)
        for(int m=0; m<M; m += 8) {
            float[] v = new float[64];
            for(int ok=0; ok<K; ok += 8) {
                for(int ik=0; ik<8; ik++) {
                    int k = ok + ik;
                    abuf[0*8 + ik] = A[(n + 0)*K + k]; 
                    abuf[1*8 + ik] = A[(n + 1)*K + k]; 
                    abuf[2*8 + ik] = A[(n + 2)*K + k]; 
                    abuf[3*8 + ik] = A[(n + 3)*K + k]; 
                    abuf[4*8 + ik] = A[(n + 4)*K + k]; 
                    abuf[5*8 + ik] = A[(n + 5)*K + k]; 
                    abuf[6*8 + ik] = A[(n + 6)*K + k]; 
                    abuf[7*8 + ik] = A[(n + 7)*K + k]; 
                }
                
                for(int ik=0; ik<8; ik++) {
                    for(int i=0; i<8; i++) a[i] = abuf[i*8 + ik];
                    
                    //System.arraycopy(B, k*M + m, b, 0,  8);
                    int k = ok + ik;
                    b[0] = B[k*M + (m + 0)];
                    b[1] = B[k*M + (m + 1)];
                    b[2] = B[k*M + (m + 2)];
                    b[3] = B[k*M + (m + 3)];
                    b[4] = B[k*M + (m + 4)];
                    b[5] = B[k*M + (m + 5)];
                    b[6] = B[k*M + (m + 6)];
                    b[7] = B[k*M + (m + 7)];
               
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
            }
            
            for(int i=0; i<8; i++)
            for(int j=0; j<8; j++)
                C[(n + i)*M + (m + j)] = v[i*8 + j];
        }
    }
    
    //Size = 1.000000, Time = 403.399994 msec, Performance = 5.323460 GFlop/s
    static void matMul5(float[] A, float[] B, float[] C, int N, int M, int K) {
        float a[] = new float[8];
        float b[] = new float[8];
        
        for(int n=0; n<N; n += 8)
        for(int m=0; m<M; m += 8) {
            float[] v = new float[64];
            for(int k=0; k<K; k++) {
                a[0] = A[(n + 0)*K + k];
                a[1] = A[(n + 1)*K + k];
                a[2] = A[(n + 2)*K + k];
                a[3] = A[(n + 3)*K + k];
                a[4] = A[(n + 4)*K + k];
                a[5] = A[(n + 5)*K + k];
                a[6] = A[(n + 6)*K + k];
                a[7] = A[(n + 7)*K + k];
                
                //System.arraycopy(B, k*M + m, b, 0,  8);
                b[0] = B[k*M + (m + 0)];
                b[1] = B[k*M + (m + 1)];
                b[2] = B[k*M + (m + 2)];
                b[3] = B[k*M + (m + 3)];
                b[4] = B[k*M + (m + 4)];
                b[5] = B[k*M + (m + 5)];
                b[6] = B[k*M + (m + 6)];
                b[7] = B[k*M + (m + 7)];
               
                
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
            
            for(int i=0; i<8; i++)
            for(int j=0; j<8; j++)
                C[(n + i)*M + (m + j)] = v[i*8 + j];
        }
    }
    
    //Size = 1.000000, Time = 489.799988 msec, Performance = 4.384409 GFlop/s
    static void matMul4(float[] A, float[] B, float[] C, int N, int M, int K) {
        float a[] = new float[4];
        float b[] = new float[4];
        
        for(int n=0; n<N; n += 4)
        for(int m=0; m<M; m += 4) {
            float[] v = new float[16];
            for(int k=0; k<K; k++) {
                a[0] = A[(n + 0)*K + k];
                a[1] = A[(n + 1)*K + k];
                a[2] = A[(n + 2)*K + k];
                a[3] = A[(n + 3)*K + k];
                
                b[0] = B[k*M + (m + 0)];
                b[1] = B[k*M + (m + 1)];
                b[2] = B[k*M + (m + 2)];
                b[3] = B[k*M + (m + 3)];
                
                v[0*4 + 0] += a[0]*b[0]; v[0*4 + 1] += a[0]*b[1]; v[0*4 + 2] += a[0]*b[2]; v[0*4 + 3] += a[0]*b[3];
                v[1*4 + 0] += a[1]*b[0]; v[1*4 + 1] += a[1]*b[1]; v[1*4 + 2] += a[1]*b[2]; v[1*4 + 3] += a[1]*b[3];
                v[2*4 + 0] += a[2]*b[0]; v[2*4 + 1] += a[2]*b[1]; v[2*4 + 2] += a[2]*b[2]; v[2*4 + 3] += a[2]*b[3];
                v[3*4 + 0] += a[3]*b[0]; v[3*4 + 1] += a[3]*b[1]; v[3*4 + 2] += a[3]*b[2]; v[3*4 + 3] += a[3]*b[3];
            }
            
            for(int i=0; i<4; i++)
            for(int j=0; j<4; j++)
                C[(n + i)*M + (m + j)] = v[i*4 + j];
        }
    }
    
    //Size = 1.000000, Time = 1288.199951 msec, Performance = 1.667042 GFlop/s
    static void matMul3(float[] A, float[] B, float[] C, int N, int M, int K) {
        float a[] = new float[4];
        float b[] = new float[4];
        
        for(int n=0; n<N; n += 4)
        for(int m=0; m<M; m += 4) {
            float[] v = new float[16];
            for(int k=0; k<K; k++) {
                for(int i=0; i<4; i++) a[i] = A[(n + i)*K + k];
                for(int i=0; i<4; i++) b[i] = B[k*M + (m + i)];
                
                for(int i=0; i<4; i++)
                for(int j=0; j<4; j++) {
                    v[i*4 + j] += a[i]*b[j];
                }
            }
            
            for(int i=0; i<4; i++)
            for(int j=0; j<4; j++)
                C[(n + i)*M + (m + j)] = v[i*4 + j];
        }
    }
    
   //Size = 1.000000, Time = 554.000000 msec, Performance = 3.876324 GFlop/s
   static void matMul2(float[] A, float[] B, float[] C, int N, int M, int K) {
        for(int n=0; n<N; n += 2)
        for(int m=0; m<M; m += 2) {
            float v00 = 0, v01 = 0;
            float v10 = 0, v11 = 0;
            for(int k=0; k<K; k++) {
                float a0 = A[n*K + k], a1 = A[(n + 1)*K + k];
                float b0 = B[k*M + m], b1 = B[k*M + m + 1];
                v00 += a0*b0; v01 += a0*b1;
                v10 += a1*b0; v11 += a1* b1;
            }
            C[n*M + m] = v00; C[n*M + m + 1] = v01;
            C[(n + 1)*M + m] = v10; C[(n + 1)*M + m + 1] = v11;
        }
    }
    
    //Size = 1.000000, Time = 2258.000000 msec, Performance = 0.951056 GFlop/s
    static void matMul1(float[] A, float[] B, float[] C, int N, int M, int K) {
        for(int m=0; m<M; m++)
        for(int n=0; n<N; n++) {
            float v = 0;
            for(int k=0; k<K; k++) v += A[n*K + k] * B[k*M + m];
            C[n *M + m] = v;
        }
    }
    
    public static void main(String[] args) {
        int N = 1024, M = 1024, K = 1024;
        
        float[] A = Vector.random_float_vector(N*K);
        float[] B = Vector.random_float_vector(K*M);
        float[] C1 = new float[N*M]; matMul1(A, B, C1, N, M, K);
        
        float[] C2 = new float[N*M]; 
        matMul7(A, B, C2, N, M, K);
        
        float sp = Vector.samePercent_absolute(C1, C2);
        System.out.println("sp = " + sp) ;
        if(sp != 1) return;
        
        int nIter = 10;
        SimpleTimer timer = SimpleTimer.clock();
        for(int i=0; i<nIter; i++) {
            matMul7(A, B, C2, N, M, K);
        }
        long dif = timer.record().timeStamp_dif_millis();
        
        System.out.println("total time = " + dif);
        float time = (float) (1.0*dif / nIter);
	double sizeV = 1.0 * N * M * K;
	float performance = (float) ((2 * sizeV * 1.0e-9f) / (time / 1000.0f));
        System.out.format("Size = %f, Time = %f msec, Performance = %f GFlop/s\n", 
                (float)(sizeV/(1024*1024*1024)), time, performance);
    }
}
