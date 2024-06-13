package test.cuda.img;


import z.dragon.engine.Engine;
import static z.dragon.alpha.Alpha.alpha;
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
public class Cuda_img_function_test_field2 
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
        
        float alpha1 = exr.nextFloat(), beta1 = exr.nextFloat();
        float alpha2 = exr.nextFloat(), beta2 = exr.nextFloat();
        float gamma = exr.nextFloat(), C = exr.nextFloat();
        
        byte[] X = Vector.random_byte_vector(length);
        float[] X1 = Vector.random_float_vector(height);
        float[] X2 = Vector.random_float_vector(height);
        
        Tensor tX = eg.tensor_int8(X, height, width);
        Tensor tX1 = eg.tensor(X1, height);
       
        //CPU-------------------------------------------------------------------
        byte[] Y1 = Vector.img_linear2_field(X, X2, alpha1, beta1, gamma, height, width);
        
        System.out.print("CPU : "); Vector.println(Y1, 0, 10);
        
        //GPU-------------------------------------------------------------------
        Tensor tY1 = eg.img.linear2_field(false, tX, tX1, alpha1, beta1, gamma);
        Tensor tY2 = eg.img.linear2_field(true, tX, tX1, alpha1, beta1, gamma);

        //compare---------------------------------------------------------------
        byte[] Y2 = tY1.raw_data();
        System.out.print("GPU1: "); Vector.println(Y2, 0, 10);
        float sp1 = Vector.samePercent_absolute(Y1, Y2); System.out.println("sp1:" + sp1);
        
        byte[] Y3 = tY2.raw_data();
        System.out.print("GPU2: "); Vector.println(Y3, 0, 10);
        float sp2 = Vector.samePercent_absolute(Y1, Y3); System.out.println("sp2:" + sp2);
        
        if(sp1 < 0.95) throw new RuntimeException();
        if(sp2 < 0.95) throw new RuntimeException();
        
        //delete----------------------------------------------------------------
        System.gc();
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
        catch(Exception e)
        {
            e.printStackTrace();
        }
        System.out.println(CudaException.lastException());
    }
}
 