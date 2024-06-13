/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package test.cuda.function;

import static z.dragon.alpha.Alpha.alpha;
import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.util.math.ExRandom;
import z.util.math.vector.Vector;

/**
 *
 * @author Gilgamesh
 */
public class Cuda_function_fusion_test1
{
    static { alpha.home("C:\\Users\\Gilgamesh\\Desktop\\Dragon-alpha-v1.1");}
    static Engine eg = alpha.engine.cuda_float32(0, alpha.engine.memp1());
    static final ExRandom exr = new ExRandom();
    
    public static void testCorrect(int height, int width) {
        int stride = (width + 3) >> 2 << 2;
	int lengthv = height * stride;
        int length = height * width;
        
        System.out.println("\nTest Corret:");
        System.out.format("(height, width, stride) = (%d, %d, %d)\n", height, width, stride);
        System.out.format("(lengthv, length) = (%d, %d)\n", lengthv, length);
        
        float[] X1 = Vector.random_float_vector(length, -1f, 1f); 
        float[] X2 = Vector.random_float_vector(length, -1f, 1f); 
        Tensor tX1 = eg.tensor(X1, height, width);
        Tensor tX2 = eg.tensor(X2, height, width);
        
        float alpha = exr.nextFloat(0, 0.5f), alpha2 = exr.nextFloat();
        float beta = exr.nextFloat(1, -1), beta2 = exr.nextFloat();
        float gamma = exr.nextFloat();
        float k = exr.nextFloat();
        
        //CPU-------------------------------------------------------------------
        float[] Y1 = Vector.linear2(X1, X2, alpha, beta, gamma);
        Y1 = Vector.leakyRelu(Y1, k);
        
        System.out.print("CPU : "); Vector.println(Y1, 0, 10);
        
        //GPU-------------------------------------------------------------------
        Tensor tY1 = eg.linear2_leakyRelu(false, tX1, tX2, alpha, beta, gamma, k);
        Tensor tY2 = eg.linear2_leakyRelu(true, tX1, tX2, alpha, beta, gamma, k);

        //compare---------------------------------------------------------------
        float[] Y2 = eg.valueOf(tY1);
        System.out.print("GPU1: "); Vector.println(Y2, 0, 10);
        float sp1 = Vector.samePercent_relative(Y1, Y2); System.out.println("sp1:" + sp1);
        if(sp1 < 0.99) throw new RuntimeException();
        
        float[] Y3 = eg.valueOf(tY2);
        System.out.print("GPU1: "); Vector.println(Y3, 0, 10);
        float sp2 = Vector.samePercent_relative(Y2, Y3); System.out.println("sp2:" + sp2);
        if(sp2 < 0.99) throw new RuntimeException();
        //delete----------------------------------------------------------------
        eg.delete(tX1, tX2, tY1);
    }
    
    public static void main(String[] args)
    {
        Vector.PRINT_DIFFERENT = true;
        
        for(int h=1; h<=10; h++)
            for(int w=1; w<=256; w++) testCorrect(h, w);
        testCorrect(1024, 1024);
    }
}
