package test.cuda.function;


import static z.dragon.alpha.Alpha.alpha;
import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.dragon.engine.cuda.impl.CudaException;
import z.util.math.ExRandom;
import z.util.math.vector.Vector;

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/**
 *
 * @author Gilgamesh
 */
public class Cuda_function_softmax_test 
{
    static { alpha.home("C:\\Users\\Gilgamesh\\Desktop\\Dragon-alpha-v1.1");}
    static Engine eg = alpha.engine.cuda_float32(0, alpha.engine.memp1());
    static final ExRandom exr = new ExRandom();
    
    public static void testCorrect(int height, int width)
    {
        int stride = (width + 3) >> 2 << 2;
	int lengthv = height * stride;
        int length = height * width;
        
        System.out.println("\nTest Corret:");
        System.out.format("(height, width, stride) = (%d, %d, %d)\n", height, width, stride);
        System.out.format("(lengthv, length) = (%d, %d)\n", lengthv, length);
        
        float[] X1 = Vector.random_float_vector(length, 0, 0.7f); Tensor tX1 = eg.tensor(X1, height, width);
        float[] X2 = Vector.random_float_vector(length, 0, 0.7f); Tensor tX2 = eg.tensor(X2, height, width);
        
        //GPU-------------------------------------------------------------------
        Tensor tY1 = eg.softmax(false, tX1); 
        tY1 = eg.crossEntropy(tY1, tX2);
        
        Tensor tY2 = eg.softmax_crossEntropy(tX1, tX2);

        //compare---------------------------------------------------------------
        float[] Y1 = tY1.value();
        float[] Y2 = tY2.value();
        
        System.out.print("GPU1: "); Vector.println(Y1, 0, 10);
        System.out.print("GPU2: "); Vector.println(Y2, 0, 10);
        
        float sp1 = Vector.samePercent_relative(Y1, Y2, 1e-4f); System.out.println("sp1:" + sp1);
        if(sp1 < 0.99) throw new RuntimeException();
        
        //delete----------------------------------------------------------------
        eg.delete(tX1, tX2, tY1, tY2);
    }
    
    public static void main(String[] args)
    {
        try
        {
            Vector.PRINT_DIFFERENT = true;
            
            for(int h=2; h<=20; h++)
                for(int w=2; w<=256; w++) testCorrect(h, w);
//            
            for(int h=100; h<=105; h++)
                for(int w= 40; w<=64; w++) testCorrect(h, w);
        
            testCorrect(1024, 1024);
        }
        catch(Exception e)
        {
            e.printStackTrace();
        }
        System.out.println(CudaException.lastException());
    }
}
