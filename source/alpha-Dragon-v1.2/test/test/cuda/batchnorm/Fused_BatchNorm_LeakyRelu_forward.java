/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package test.cuda.batchnorm;

import static z.dragon.alpha.Alpha.alpha;
import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.util.math.ExRandom;
import z.util.math.vector.Vector;

/**
 *
 * @author Gilgamesh
 */
public class Fused_BatchNorm_LeakyRelu_forward 
{
    static { alpha.home("C:\\Users\\Gilgamesh\\Desktop\\Dragon-alpha-v1.1");}
    static Engine eg = alpha.engine.cuda_float32(0, alpha.engine.memp1());
    static final ExRandom exr = new ExRandom();
    
    //affine = false
    public static void testCorrect(int height, int width)
    {
        System.out.format("testCorrect: (height, width) = (%d, %d)\n", height, width);
        
        float eps = exr.nextFloat();
        float k = exr.nextFloat();
        Tensor A = eg.Gaussian(width);
        Tensor B = eg.Gaussian(width);
        
        Tensor X = eg.uniform(-1, 1, height, width);
        Tensor[] stats = eg.field_var_mean(false, X);
        Tensor X_var = stats[0];
        Tensor X_mean = stats[1];
       
        //path1-----------------------------------------------------------------
//        Tensor tY1 = eg.batchNorm(false, X, X_mean, X_var, eps).leakyRelu(true, k);
        Tensor tY1 = eg.batchNorm(false, X, X_mean, X_var, eps, A, B).leakyRelu(true, k);
        
        //path2-----------------------------------------------------------------
//        Tensor tY2 = eg.batchNorm_leakyRelu(false, X, X_mean, X_var, eps, k);
        Tensor tY2 = eg.batchNorm_leakyRelu(true, X, X_mean, X_var, eps, A, B, k);

        //compare---------------------------------------------------------------
        float[] Y1 = tY1.value();
        float[] Y2 = tY2.value();
        float sp = Vector.samePercent_relative(Y1, Y2);
        
        Vector.println("Y1 = ", Y1, 0, 10);
        Vector.println("Y2 = ", Y2, 0, 10);
        System.out.println("sp = " + sp);
        if(sp < 0.99f) throw new RuntimeException();
        System.gc();
    }
    
    
    public static void main(String[] args)
    {
        for(int h=2; h<=32; h++)
            for(int w=1; w<=256; w++) testCorrect(h, w);
            
        for(int h=100; h<=105; h++)
            for(int w= 128; w<=256; w++) testCorrect(h, w);
            
        for(int h=1000; h<=1028; h++)
            for(int w= 233; w<=256; w++) testCorrect(h, w);
    }
}
