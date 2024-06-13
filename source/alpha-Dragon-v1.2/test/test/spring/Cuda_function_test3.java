/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package test.spring;

import static z.dragon.alpha.Alpha.UnitFunctional.F;
import static z.dragon.alpha.Alpha.alpha;
import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.dragon.engine.cuda.impl.Cuda_expk2;
import z.util.math.ExRandom;
import z.util.math.vector.Vector;

/**
 *
 * @author Gilgamesh
 */
public class Cuda_function_test3 
{
    static final ExRandom exr = new ExRandom();
    static { alpha.home("C:\\Users\\Gilgamesh\\Desktop\\Dragon-alpha");}
    static Engine eg = alpha.engine.cuda_float32(0, alpha.engine.memp1());
    
    static float[][] matMul(float[][] A, float[][] B) {
        int N = A.length, M = B[0].length, K = B.length;
        float[][] C = new float[N][M];
        for(int k = 0; k<K; k++)
            for(int i=0; i<N; i++)
                for(int j=0; j<M; j++)
                    C[i][j] += A[i][k] * B[k][j];
        return C;
    }
    
    public static void testCorrect(int height, int width)
    {
        int stride = (width + 3) >> 2 << 2;
	int lengthv = height * stride;
        int length = height * width;
        
        System.out.println("\nTest Corret:");
        System.out.format("(height, width, stride) = (%d, %d, %d)\n", height, width, stride);
        System.out.format("(lengthv, length) = (%d, %d)\n", lengthv, length);
        
        float[] X = Vector.random_float_vector(height * width, -1, 1f);
        float[] W = Vector.random_float_vector(width * width, -1, 1f);
        
        Tensor tX = eg.tensor(X , height, width).c();
        Tensor tW = eg.tensor(W, width, width).c();
        
        float alpha = exr.nextFloat(0, 0.5f), alpha2 = exr.nextFloat();
        float beta = exr.nextFloat(0, 0.5f), beta2 = exr.nextFloat();
        float gamma = exr.nextFloat();
        
        float vmin = exr.nextFloat(0, -1.5f);
        float vmax = exr.nextFloat(0, 1.5f);
        float k = exr.nextFloat();
        
        System.out.println("k = " + k);
        System.out.println("alpha, beta = " + alpha + ", " + beta);
        
        //CPU-------------------------------------------------------------------
        float[][] mX = Vector.to2D(X, height, width);
        float[][] mW = Vector.to2D(W, width, width);
        float[][] mY = matMul(mX, mW);
        float[] Y = Vector.flatten(mY);
        
        float[] Y1 = Vector.leakyRelu(Y, k);
//        float[] Y1 = Vector.softplus(X);
//        float[] Y1 = Vector.elu(X, 1.0f, 0.01f);
//        float[] Y1 = Vector.elu(X, alpha, k);
        
//        float[] Y1 = Vector.sigmoid(X);
//        float[] Y1 = Vector.tanh(X);
    
//        float[] Y1 = Vector.cot(X);
//        float[] Y1 = Vector.cot(alpha, X, beta);
        
//        float[] Y1 = Vector.arcsin(X);
//        float[] Y1 = Vector.arcsin(alpha, X, beta);
//        float[] Y1 = Vector.arctan(X);
//        float[] Y1 = Vector.arctan(alpha, X, beta);

//        float[] Y1 = Vector.abs(X);
//        float[] Y1 = Vector.abs(alpha, X, beta);

//        float[] Y1 = Vector.sin(X);
//        float[] Y1 = Vector.sin(alpha, X, beta);
//        float[] Y1 = Vector.cos(X);
//        float[] Y1 = Vector.cos(alpha, X, beta);

//        float[] Y1 = Vector.square(X);
//        float[] Y1 = Vector.square(alpha, X);
//        float[] Y1 = Vector.quadratic(X, alpha, beta, gamma);


        System.out.print("CPU : "); Vector.println(Y1, 0, 10);
        
        //GPU-------------------------------------------------------------------
        Tensor tY = eg.matMul(tX, tW).c();
        
        Tensor tY1 = F.leakyRelu(k, tY)[0];
//        Tensor tY1 = F.softplus(tX)[0];
//        Tensor tY1 = F.elu(tX)[0];
//        Tensor tY1 = F.elu(alpha, k, tX)[0];

//        Tensor tY1 = F.sigmoid(tX)[0];
//        Tensor tY1 = F.tanh(tX)[0];
//        Tensor tY1 = F.softmax(width, tX)[0];
//        Tensor tY1 = F.log_softmax(width, tX)[0];
    
//        Tensor tY1 = F.tan(tX)[0];
//        Tensor tY1 = F.tan(alpha, beta, tX)[0];
//        Tensor tY1 = F.cot(tX)[0]; 
//        Tensor tY1 = F.cot(alpha, beta, tX)[0];

//        Tensor tY1 = F.arcsin(tX)[0];
//        Tensor tY1 = F.arcsin(alpha, beta, tX)[0];
//        Tensor tY1 = F.arctan(tX)[0];
//        Tensor tY1 = F.arctan(alpha, beta, tX)[0];
        
//        Tensor tY1 = F.abs(tX)[0];
//        Tensor tY1 = F.abs(alpha, beta, tX)[0];

//        Tensor tY1 = F.sin(tX)[0];
//        Tensor tY1 = F.sin(alpha, beta, tX)[0];
//        Tensor tY1 = F.cos(tX)[0];
//        Tensor tY1 = F.cos(alpha, beta, tX)[0];
    
//        Tensor tY1 = F.square(tX)[0];
//        Tensor tY1 = F.square(alpha, tX)[0];
//        Tensor tY1 = F.quadratic(alpha, beta, gamma, tX)[0];

        //compare---------------------------------------------------------------
        
        float[] Y2 = eg.valueOf(tY1);
        System.out.print("GPU1: "); Vector.println(Y2, 0, 10);
        float sp1 = Vector.samePercent_relative(Y1, Y2); System.out.println("sp1:" + sp1);
        if(sp1 < 0.99) throw new RuntimeException();
        
        //delete----------------------------------------------------------------
        Cuda_expk2.checkMemAlign(tX);
        Cuda_expk2.checkMemAlign(tY1);
        tX.delete();
        tY.delete();
        tY1.delete();
    }
    
    public static void main(String[] args)
    {
        for(int h=1; h<=64; h++)
        for(int w=1; w<=64; w++)
            testCorrect(h, w);
    }
}
