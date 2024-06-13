/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package test.cuda.reduce;

import static z.dragon.alpha.Alpha.alpha;
import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.dragon.engine.cuda.CudaFloat32EngineBase;
import z.dragon.engine.cuda.impl.CudaException;
import z.util.math.ExRandom;
import z.util.math.vector.Matrix;
import z.util.math.vector.Vector;

/**
 *
 * @author Gilgamesh
 */
public class Cuda_reduce_field_test1 
{
    static { alpha.home("C:\\Users\\Gilgamesh\\Desktop\\Dragon-alpha-v1.1");}
    static Engine eg = alpha.engine.cuda_float32(0, alpha.engine.memp1());
    static final ExRandom exr = new ExRandom(); 
    static {
        CudaFloat32EngineBase cu32 = (CudaFloat32EngineBase) eg.engineBase();
        //safe = false: std: correct, var: correct
        //safe = true:  std:          var: correct
        cu32.field_var_safe(false);
        cu32.field_std_safe(false);
    }

    public static void testCorrect(int height, int width)
    {
        int stride = (width + 3) >> 2 << 2;
	int lengthv = height * stride;
        int length  = height * width;
        
        System.out.println("\nTest Corret:");
        System.out.format("(height, width, stride) = (%d, %d, %d)\n", height, width, stride);
        System.out.format("(lengthv, length) = (%d, %d)\n", lengthv, length);
        
        float alpha1 = exr.nextFloat(), beta1 = exr.nextFloat();
        float alpha2 = exr.nextFloat(), beta2 = exr.nextFloat();
        float gamma = exr.nextFloat();
        boolean unbiased = true;
        
        float[] X = Vector.random_float_vector(length, -2f, 3f);
        Tensor tX = eg.tensor(X, height, width);
        
        //CPU-------------------------------------------------------------------
        float[][] mX = Matrix.toMatrix(X, width);
        
//        float[][] Ys = Matrix.field_var(unbiased, mX);
        float[][] Ys = Matrix.field_std(unbiased, mX);
        
        float[] var1 = Ys[0];
        float[] mean1 = Ys[1];
        
        //GPU-------------------------------------------------------------------
//        Tensor[] tYs = eg.field_var_mean(unbiased, tX, width);
        Tensor[] tYs = eg.field_std_mean(unbiased, tX, width);
        
        float[] var2 = tYs[0].value();
        float[] mean2 = tYs[1].value();
        
        //compare---------------------------------------------------------------
        
        float sp1 = Vector.samePercent_relative(var1, var2);
        float sp2 = Vector.samePercent_relative(mean1, mean2);
        
        System.out.println("sp(var) = " + sp1);
        Vector.println("var1 = ", var1, 0, 10);
        Vector.println("var2 = ", var2, 0, 10);
        
        System.out.println("sp(mean) = " + sp2);
        Vector.println("mean1 = ", mean1, 0, 10);
        Vector.println("mean2 = ", mean2, 0, 10);
        
        if(sp1 < 0.90) throw new RuntimeException();
        if(sp2 < 0.90) throw new RuntimeException();
        
        if(Float.isNaN(eg.straight_sum(tYs[0]).get())) throw new RuntimeException();
      
        eg.delete(tYs);
        eg.delete(tX);
    }
  
    public static void main(String[] args)
    {
        try
        {
             Vector.PRINT_DIFFERENT = true;
            for(int h=5; h<=43; h++)
                for(int w=2; w<=256; w++) testCorrect(h, w);
////            
           
//            for(int h=100; h<=105; h++)
//                for(int w= 128; w<=256; w++) testCorrect(h, w);
        
//            testSpeed(1024, 1024);
        }
        catch(Exception e)
        {
            e.printStackTrace();
        }
        System.out.println(CudaException.lastException());
    }
}
