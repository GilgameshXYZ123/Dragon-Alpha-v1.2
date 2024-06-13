/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package test.cuda;

import static test.cuda.function.Cuda_function_test1.testCorrect;
import static z.dragon.alpha.Alpha.UnitBuilder.nn;
import static z.dragon.alpha.Alpha.UnitFunctional.F;
import static z.dragon.alpha.Alpha.alpha;
import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.dragon.nn.unit.Unit;
import z.util.math.ExRandom;
import z.util.math.vector.Vector;

/**
 *
 * @author Gilgamesh
 */
public class Cuda_bernouli_leaky_relu_test 
{
    static { alpha.home("C:\\Users\\Gilgamesh\\Desktop\\Dragon-alpha-v1.2");}
    static Engine eg = alpha.engine.cuda_float32(0, alpha.engine.memp1());
    static ExRandom exr = new ExRandom();
    
    public static void testCorrect(int height, int width) {
        int stride = (width + 3) >> 2 << 2;
	int lengthv = height * stride;
        int length = height * width;
        long seed = 10022222L;
        
        System.out.println("\nTest Corret:");
        System.out.format("(height, width, stride) = (%d, %d, %d)\n", height, width, stride);
        System.out.format("(lengthv, length) = (%d, %d)\n", lengthv, length);
        
        float p = exr.nextFloat();
        float v1 = exr.nextFloat(-1.0f, 1.0f);
        float v2 = exr.nextFloat(-1.0f, 1.0f);
        float k = exr.nextFloat(); System.out.println(k);
        
        //GPU-------------------------------------------------------------------
        Unit leaky_berm = nn.leakyRelu_bernouliMul(
                nn.leakyRelu(k),
                nn.bernouliMul(p, v1, v2)).inplace(false);
        
        Unit berm = nn.bernouliMul(p, v1, v2).inplace(false);
        
        Tensor X = eg.Gaussian(height, width).need_grad(true);
        
        eg.set_seed(seed); Tensor Y1 = leaky_berm.forward(X)[0];
        eg.set_seed(seed); Tensor Y2 = berm.forward(F.leakyRelu(k, X))[0]; 
        
        Tensor deltaY1 = eg.set_seed(seed).Gaussian(height, width);
        Tensor deltaY2 = eg.set_seed(seed).Gaussian(height, width);
        
        Tensor deltaX1 = leaky_berm.backward(deltaY1)[0];
        Tensor deltaX2 = berm.backward(deltaY2)[0];
        deltaX2 = eg.leakyRelu_deltaX_v2(true, deltaX2, X, k);
        
        //compare---------------------------------------------------------------
        float[] y1 = Y1.value(); Vector.println("Y1 = ", y1, 0, 10);
        float[] y2 = Y2.value(); Vector.println("Y2 = ", y2, 0, 10);
        
        float[] dX1 = deltaX1.value(); Vector.println("dX1 = ", dX1, 0, 10);
        float[] dX2 = deltaX2.value(); Vector.println("dX2 = ", dX2, 0, 10);
        
        float sp1 = Vector.samePercent_relative(y1, y2); System.out.println("sp1:" + sp1);
        if(sp1 < 0.99) throw new RuntimeException();
        
        float sp2 = Vector.samePercent_relative(dX1, dX2); System.out.println("sp2:" + sp2);
        if(sp2 < 0.99) throw new RuntimeException();
        
        System.gc();
    }
    
    public static void main(String[] args)
    {
        for(int h = 1; h <= 20; h++)
                for(int w = 1; w <= 256; w++) testCorrect(h, w);
//            
            for(int h=100; h<=105; h++)
                for(int w= 40; w<=64; w++) testCorrect(h, w);
            
            for(int h=300; h<=305; h++)
                for(int w=7; w<=12; w++) testCorrect(h, w);
            
            for(int h=300; h<=305; h++)
                for(int w= 450; w<=512; w++) testCorrect(h, w);
    }
    
}
