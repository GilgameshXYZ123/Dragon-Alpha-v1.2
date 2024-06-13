/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package test.cuda.function;

import static z.dragon.alpha.Alpha.alpha;
import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.dragon.engine.cuda.impl.CudaException;
import z.util.math.ExRandom;
import z.util.math.vector.Vector;

/**
 *
 * @author Gilgamesh
 */
public class Cuda_function_testB0 
{
    static final ExRandom exr = new ExRandom();
    static { alpha.home("C:\\Users\\Gilgamesh\\Desktop\\Dragon-alpha-v1.1");}
    static Engine eg = alpha.engine.cuda_float32(0, alpha.engine.memp1());
    
    public static void testCorrect(int height, int width)
    {
        int stride = (width + 3) >> 2 << 2;
	int lengthv = height * stride;
        int length = height * width;
        
        System.out.println("\nTest Corret:");
        System.out.format("(height, width, stride) = (%d, %d, %d)\n", height, width, stride);
        System.out.format("(lengthv, length) = (%d, %d)\n", lengthv, length);

        float alpha = exr.nextFloat(), beta = exr.nextFloat();
        
        byte[] BX = Vector.random_byte_vector(length);
        Tensor tBX = eg.tensor_int8(BX, height, width);
        
        //CPU-------------------------------------------------------------------
//        float[] Y1 = Vector.pix2tensor(BX);
        float[] Y1 = Vector.linear_int8_to_float(alpha, BX, beta); 
        
        System.out.print("CPU : "); Vector.println(Y1, 0, 10);
        
        //GPU-------------------------------------------------------------------
//        Tensor tY1 = eg.pix2tensor(false, tBX);
//        Tensor tY2 = eg.pix2tensor(true, tBX);
        Tensor tY1 = eg.linear_int8_to_dtype(false, alpha, tBX, beta).c();
        Tensor tY2 = eg.linear_int8_to_dtype(true, alpha, tBX, beta).c();

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
        eg.delete(tBX, tY2, tY1);
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
        catch(Exception e) {
            e.printStackTrace();
        }
        System.out.println(CudaException.lastException());
    }
}
