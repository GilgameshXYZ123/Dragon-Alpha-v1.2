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
public class Cuda_function_deltaX_test2
{
    static { alpha.home("C:\\Users\\Gilgamesh\\Desktop\\Dragon-alpha-v1.2");}
    static Engine eg = alpha.engine.cuda_float32(0, alpha.engine.memp1());
    static final ExRandom exr = new ExRandom();
    
    public static void testCorrect(int height, int width) {
        int stride = (width + 3) >> 2 << 2;
	int lengthv = height * stride;
        int length = height * width;
        
        System.out.println("\nTest Corret:");
        System.out.format("(height, width, stride) = (%d, %d, %d)\n", height, width, stride);
        System.out.format("(lengthv, length) = (%d, %d)\n", lengthv, length);
        
        float alpha = exr.nextFloat();
        float beta = exr.nextFloat();
        float vmax = exr.nextFloat(1, 2);
        System.out.println("vmax = " + vmax);
        
        Tensor X = eg.Gaussian(height, width);
        Tensor Y = eg.max(false, alpha, X, beta, vmax);
        Tensor deltaY = eg.Gaussian(height, width);
        
        Tensor deltaX1 = eg.max_deltaX_v1(false, deltaY, Y, alpha, vmax);
        Tensor deltaX2 = eg.max_deltaX_v2(false, deltaY, X, alpha, beta, vmax);
        
        float sp = deltaX1.equal(deltaX2).get();
        System.out.println("sp = " + sp);
        System.out.println("Y.max = " + Y.min());
        System.gc();
    }
    
    public static void main(String[] args)
    {
        Vector.PRINT_DIFFERENT = true;
        
        for(int h=1; h<=10; h++)
            for(int w=1; w<=256; w++) testCorrect(h, w);
        testCorrect(1024, 1024);
    }
}
