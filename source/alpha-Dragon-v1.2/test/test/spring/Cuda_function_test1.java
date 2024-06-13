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
import z.dragon.engine.cuda.impl.CudaException;
import z.dragon.engine.cuda.impl.Cuda_expk2;
import z.util.math.ExRandom;
import z.util.math.vector.Vector;

/**
 *
 * @author Gilgamesh
 */
public class Cuda_function_test1 
{
    static final ExRandom exr = new ExRandom();
    static { alpha.home("C:\\Users\\Gilgamesh\\Desktop\\Dragon-alpha");}
    static Engine eg = alpha.engine.cuda_float32(0, alpha.engine.memp1());
    
    public static void testCorrect(int height, int width)
    {
        eg.sync(false);
        
        int stride = (width + 3) >> 2 << 2;
	int lengthv = height * stride;
        int length = height * width;
        
        System.out.println("\nTest Corret:");
        System.out.format("(height, width, stride) = (%d, %d, %d)\n", height, width, stride);
        System.out.format("(lengthv, length) = (%d, %d)\n", lengthv, length);
        
        float[] X = Vector.random_float_vector(length, 0, 1f);
        Tensor tX  = eg.tensor(X , height, width).c();
        
        float alpha = exr.nextFloat(0, 0.5f), alpha2 = exr.nextFloat();
        float beta = exr.nextFloat(0, 0.5f), beta2 = exr.nextFloat();
        float gamma = exr.nextFloat();
        
        float vmin = exr.nextFloat(0, -1.5f);
        float vmax = exr.nextFloat(0, 1.5f);
        float k = exr.nextFloat();
        
        Vector.println("X:" , X, 0, 10);
        System.out.println("k = " + k);
        System.out.println("alpha, beta = " + alpha + ", " + beta);
        
        //CPU-------------------------------------------------------------------
//        float[] Y1 = Vector.sadd(X, k);
//        float[] Y1 = Vector.ssub(X, k);
//        float[] Y1 = Vector.smul(X, k);
//        float[] Y1 = Vector.sdiv(X, k);
//        float[] Y1 = Vector.linear(alpha, X, beta);
        
//        float[] Y1 = Vector.rpl(X);
//        float[] Y1 = Vector.rpl(alpha, X, beta, gamma);
    
//        float[] Y1 = Vector.exp(X);
//        float[] Y1 = Vector.exp(alpha, X, beta);
            
//        alpha = Math.abs(alpha);
//        beta = Math.abs(beta);
//        float[] Y1 = Vector.sqrt(X);
//        float[] Y1 = Vector.sqrt(alpha, X, beta);
        
        alpha = Math.abs(alpha);
        beta = Math.abs(beta);
        float[] Y1 = Vector.log(X);
//        float[] Y1 = Vector.log(alpha, X, beta);
        
//        float[] Y1 = Vector.relu(X);
    
//        float[] Y1 = Vector.leakyRelu(X, 0.01f);
//        float[] Y1 = Vector.leakyRelu(X, k);
//        float[] Y1 = Vector.softplus(X);
//        float[] Y1 = Vector.elu(X, 1.0f, 0.01f);
//        float[] Y1 = Vector.elu(X, alpha, k);
        
//        float[] Y1 = Vector.sigmoid(X);
//        float[] Y1 = Vector.tanh(X);
    
//        float[][] mX = Vector.toTense2D(X, height, width);
//        float[][] mY = Matrix.softmax(mX);
//        float[] Y1 = Vector.flatten(mY);
//        Y1 = Vector.log(Y1);

//        float[] Y1 = Vector.tan(X);
//        float[] Y1 = Vector.tan(alpha, X, beta);
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
//        Tensor tY1 = F.sadd(k, tX)[0];
//        Tensor tY1 = F.ssub(k, tX)[0];
//        Tensor tY1 = F.smul(k, tX)[0];
//        Tensor tY1 = F.sdiv(k, tX)[0];
//        Tensor tY1 = F.linear(false, alpha, beta, tX)[0];

//        Tensor tY1 = F.rpl(tX)[0];
//        Tensor tY1 = F.rpl(alpha, beta, gamma, tX)[0];
            
//        Tensor tY1 = F.exp(tX)[0];
//        Tensor tY1 = F.exp(alpha, beta, tX)[0];

        Tensor tY1 = F.log(tX)[0];
//        Tensor tY1 = F.log(alpha, beta, tX)[0];

//        Tensor tY1 = F.sqrt(tX)[0];
//        Tensor tY1 = F.sqrt(alpha, beta, tX)[0];

//        Tensor tY1 = F.relu(tX)[0];
//        Tensor tY1 = F.leakyRelu(tX)[0];
//        Tensor tY1 = F.leakyRelu(k, tX)[0];
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
        System.out.println(tY1.syncer().getClass());
        System.out.println(eg.straight_sum(tY1).get());
        
        float[] Y2 = eg.valueOf(tY1);
        System.out.print("GPU1: "); Vector.println(Y2, 0, 10);
        float sp1 = Vector.samePercent_relative(Y1, Y2); System.out.println("sp1:" + sp1);
        if(sp1 < 0.99) throw new RuntimeException();
        
        //delete----------------------------------------------------------------
        Cuda_expk2.checkMemAlign(tX);
        Cuda_expk2.checkMemAlign(tY1);
        eg.delete(tX, tY1);
        
    }
    
    public static void main(String[] args)
    {
        try
        {
            Vector.PRINT_DIFFERENT = true;
            for(int h=1; h<=10; h++)
                for(int w=1; w<=256; w++) testCorrect(h, w);
            testCorrect(1024, 1024);
        }
        catch(Exception e)
        {
            e.printStackTrace();
        }
        System.out.println(CudaException.lastException());
    }
    
    
}
