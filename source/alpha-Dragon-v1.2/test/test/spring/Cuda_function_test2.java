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
public class Cuda_function_test2 
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
        
        float[] X1 = Vector.random_float_vector(length, -1, 1f);
        float[] X2 = Vector.random_float_vector(length,  0, 1f); 
        
        Tensor tX1  = eg.tensor(X1 , height, width).c();
        Tensor tX2 = eg.tensor(X2, height, width).c();
        
        float alpha = exr.nextFloat(0, 0.5f), alpha2 = exr.nextFloat();
        float beta = exr.nextFloat(0, 0.5f), beta2 = exr.nextFloat();
        float gamma = exr.nextFloat();
        
        float vmin = exr.nextFloat(0, -1.5f);
        float vmax = exr.nextFloat(0, 1.5f);
        float k = exr.nextFloat();
        
        Vector.println("X:" , X1, 0, 10);
        System.out.println("k = " + k);
        System.out.println("alpha, beta = " + alpha + ", " + beta);
        
        //CPU-------------------------------------------------------------------
//        float[] Y1 = Vector.add(X1, X2);
//        float[] Y1 = Vector.sub(X1, X2);
//        float[] Y1 = Vector.add(alpha, X1, beta, X2);
//        float[] Y1 = Vector.linear2(X1, X2, alpha, beta, gamma);
        
//        float[] Y1 = Vector.mul(X1, X2);
//        float[] Y1 = Vector.mul(alpha, X1, X2);
//        float[] Y1 = Vector.squareAdd(X1, X2);
//        float[] Y1 = Vector.squareSub(X1, X2);
//        float[] Y1 = Vector.squareAdd(alpha, X1, beta, X2);
        float[] Y1 = Vector.quadratic2(X1, X2, alpha, alpha2, beta, beta2, gamma, k);
        
        System.out.print("CPU : "); Vector.println(Y1, 0, 10);
        
        //GPU-------------------------------------------------------------------
//        Tensor tY = F.add(tX1, tX2)[0];
//        Tensor tY = F.sub(tX1, tX2)[0];
//        Tensor tY = F.add(alpha, beta, tX1, tX2)[0];
//        Tensor tY = F.linear2(alpha, beta, gamma, tX1, tX2)[0];
        
//        Tensor tY = F.mul(tX1, tX2)[0];
//        Tensor tY = F.mul(alpha, tX1, tX2)[0];
//        Tensor tY = F.squareAdd(tX1, tX2)[0];
//        Tensor tY = F.squareSub(tX1, tX2)[0];
//        Tensor tY = F.squareAdd(alpha, beta, tX1, tX2)[0];
        Tensor tY = F.quadratic2(alpha, alpha2, beta, beta2, gamma, k, tX1, tX2)[0];

        //compare---------------------------------------------------------------
        System.out.println(tY.syncer().getClass());
        
        
        float[] Y2 = eg.valueOf(tY);
        System.out.print("GPU1: "); Vector.println(Y2, 0, 10);
        float sp1 = Vector.samePercent_relative(Y1, Y2); System.out.println("sp1:" + sp1);
        if(sp1 < 0.99) throw new RuntimeException();
        
        //delete----------------------------------------------------------------
        Cuda_expk2.checkMemAlign(tX1);
        Cuda_expk2.checkMemAlign(tY);
        eg.delete(tX1, tX2, tY);
        
    }
    
    public static void main(String[] args)
    {
        try
        {
            Vector.PRINT_DIFFERENT = true;
            //0.1601053
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
