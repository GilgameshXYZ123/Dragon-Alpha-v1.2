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
public class Cuda_function_testB1 
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
        
        float[] X = Vector.random_float_vector(length, 0, 1);
        Tensor tX = eg.tensor(X, height, width);
        
        //CPU-------------------------------------------------------------------
        byte[] Y1 = Vector.tensor2pix(X);
//        byte[] Y1 = Vector.linear_float_to_int8(alpha, X, beta);
        System.out.print("CPU : "); Vector.println(Y1, 0, 10);
        
        //GPU-------------------------------------------------------------------
        Tensor tY1 = eg.tensor_to_pixel(false, tX);
        Tensor tY2 = eg.tensor_to_pixel(true, tX);
//        Tensor tY1 = eg.linear_dtype_to_int8(false, alpha, tX, beta);
//        Tensor tY2 = eg.linear_dtype_to_int8(true, alpha, tX, beta);

        //compare---------------------------------------------------------------
        byte[] Y2 = tY1.value_int8();
        System.out.print("GPU1: "); Vector.println(Y2, 0, 10);
        float sp1 = Vector.samePercent_absolute(Y1, Y2); System.out.println("sp1:" + sp1);
        
        byte[] Y3 = tY2.value_int8();
        System.out.print("GPU2: "); Vector.println(Y3, 0, 10);
        float sp2 = Vector.samePercent_absolute(Y1, Y3); System.out.println("sp2:" + sp2);
        
        //float zp1 = Vector.zeroPercent(Y2); System.out.println("zp1: " + zp1);
        //float zp2 = Vector.zeroPercent(Y3); System.out.println("zp2: " + zp2);
        
        if(sp1 < 0.99) throw new RuntimeException();
        if(sp2 < 0.99) throw new RuntimeException();
        //delete----------------------------------------------------------------
        eg.delete(tX, tY2, tY1);
    }
 
    public static void main(String[] args)
    {
        try
        {
//            Vector.PRINT_DIFFERENT = true;
            
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
