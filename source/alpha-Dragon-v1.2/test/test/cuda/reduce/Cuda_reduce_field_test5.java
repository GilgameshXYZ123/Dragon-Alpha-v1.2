/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package test.cuda.reduce;

import static z.dragon.alpha.Alpha.alpha;
import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.dragon.engine.cuda.impl.CudaException;
import z.util.math.ExRandom;
import z.util.math.vector.Matrix;
import z.util.math.vector.Vector;

/**
 *
 * @author Gilgamesh
 */
public class Cuda_reduce_field_test5 
{
    static { alpha.home("C:\\Users\\Gilgamesh\\Desktop\\Dragon-alpha-v1.1");}
    static Engine eg = alpha.engine.cuda_float32(0, alpha.engine.memp1());
    static final ExRandom exr = new ExRandom();

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
        
        gamma = 0; beta1 = 0;
        
        float[] X = Vector.random_float_vector(length, -2f, 3f);
        Tensor tX = eg.tensor(X, height, width);
        
        //CPU-------------------------------------------------------------------
        float[][] mX = Matrix.toMatrix(X, width);
        
        float[][] Ys = Matrix.field_linear_quadratic(mX, alpha1, beta1, alpha2, beta2, gamma);
        
        
        float[] V1 = Ys[0];
        float[] V2 = Ys[1];
        
        //GPU-------------------------------------------------------------------
        Tensor[] tYs = eg.field_linear_quadratic(tX, alpha1, beta1, alpha2, beta2, gamma);
        
        float[] V3 = tYs[0].value();
        float[] V4 = tYs[1].value();
        
        //compare---------------------------------------------------------------
        
        float sp1 = Vector.samePercent_relative(V1, V3);
        float sp2 = Vector.samePercent_relative(V2, V4);
        
        System.out.println("sp1 = " + sp1);
        Vector.println("V1 = ", V1, 0, 10);
        Vector.println("V3 = ", V3, 0, 10);
        
        System.out.println("sp2 = " + sp2);
        Vector.println("V2 = ", V2, 0, 10);
        Vector.println("V4 = ", V4, 0, 10);
        
        if(sp1 < 0.98) throw new RuntimeException();
        if(sp2 < 0.98) throw new RuntimeException();
        
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
                for(int w=1; w<=256; w++) testCorrect(h, w);
////            
           
            for(int h=100; h<=105; h++)
                for(int w= 128; w<=256; w++) testCorrect(h, w);
        
//            testSpeed(1024, 1024);
        }
        catch(Exception e)
        {
            e.printStackTrace();
        }
        System.out.println(CudaException.lastException());
    }
}
