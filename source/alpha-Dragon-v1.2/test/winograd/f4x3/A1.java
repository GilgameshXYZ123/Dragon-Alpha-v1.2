/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package winograd.f4x3;

import z.util.math.vector.Matrix;

/**
 *
 * @author Gilgamesh
 */
public class A1 
{
    //h = A^T * [G*p <*> B^T*s]
    
    static float[][] B = {
        {      1,       0,       0,        0,        0,  0},
        {      0,  2.0f/3, -2.0f/3, -1.0f/12,  1.0f/12,  4},
        {-5.0f/4,  2.0f/3,  2.0f/3, -1.0f/24, -1.0f/24,  0},
        {      0, -1.0f/6,  1.0f/6,  1.0f/12, -1.0f/12, -5},
        { 1.0f/4, -1.0f/6, -1.0f/6,  1.0f/24,  1.0f/24,  0},
        {      0,       0,       0,        0,      0,    1}
    };
    
    //B^t / 24
    static float[][] Bt = {//24 * B^T * s
        //0    1    2    3   4   5
        {24,   0, -30,   0,  6 , 0},//24*s0 - 30*s2 + 6*s4
        { 0,  16,  16,  -4, -4,  0},// 16*s1 + 16*s2 - 4*s3 - 4*s4
        { 0, -16,  16,   4, -4,  0},//-16*s1 + 16*s2 - 4*s3 - 4*s4
        { 0,  -2,  -1,   2,  1,  0},//-2*s1 - s2 + 2*s3 + s4 
        { 0,   2,  -1,  -2,  1,  0},// 2*s1 - s2 - 2*s3 + s4
        { 0,  96,   0, -120, 0, 24}//96*s1 - 120*s3 + 24*s5
    };
    
    static float[][] G = {//G*p = 
        {1,  0, 0}, //p0
        {1,  1, 1}, //p0 + p1 + p2
        {1, -1, 1}, //p0 - p1 + p2
        {1,  2, 4}, //p0 + 2*p1 + 4*p2
        {1, -2, 4}, //p0 - 2*p1 + 4*p2
        {0,  0, 1}  //p2
    };
    
    
    static void mul(float[][] A, float v) {
        int N = A.length, M = A[0].length;
        for(int i=0; i<N; i++)
        for(int j=0; j<M; j++) A[i][j] *= v;
    }
    
    static float[][] transpose(float[][] A){
        int N = A.length, M = A[0].length;
        float[][] B = new float[M][N];
        for(int i=0; i<N; i++)
        for(int j=0; j<M; j++) B[j][i] = A[i][j];
        return B;
    }
    
    public static void main(String[] args) {
        float[][] Bt = transpose(B);
        mul(Bt, 24);
        Matrix.println(Bt);
        
    }
    
}
