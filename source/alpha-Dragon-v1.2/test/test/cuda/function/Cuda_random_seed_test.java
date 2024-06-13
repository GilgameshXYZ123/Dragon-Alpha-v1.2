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
public class Cuda_random_seed_test 
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
        
        long seed = 123456789L;
        
        Tensor X = eg.set_seed(seed).Gaussian(height, width);
        Tensor Y = eg.set_seed(seed).Gaussian(height, width); 
        
        float[] v1 = X.value(); Vector.println("v1 = ", v1, 0, 10);
        float[] v2 = Y.value(); Vector.println("v2 = ", v2, 0, 10);
        float sp = Vector.samePercent_absolute(v1, v2);
        System.out.println("sp = " + sp);
        
        if(sp < 0.99f) throw new RuntimeException();
    }
    
    public static void main(String[] args)
    {
        int x = 512*4*4

                ;
        System.out.println(x);
        
//          for(int h = 1; h <= 20; h++)
//                for(int w = 1; w <= 256; w++) testCorrect(h, w);
//            
//            for(int h=100; h<=105; h++)
//                for(int w= 40; w<=64; w++) testCorrect(h, w);
//            
//            for(int h=300; h<=305; h++)
//                for(int w=7; w<=12; w++) testCorrect(h, w);
//            
//            for(int h=300; h<=305; h++)
//                for(int w= 450; w<=512; w++) testCorrect(h, w);
    }
}
