package test.cuda.img;


import z.dragon.engine.Engine;
import static z.dragon.alpha.Alpha.alpha;
import z.dragon.engine.Tensor;
import z.dragon.engine.cuda.impl.Cuda;
import z.dragon.engine.cuda.impl.CudaException;
import z.util.lang.SimpleTimer;
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
public class Cuda_img_function_test1 
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
        
        float alpha = exr.nextFloat();
        float beta = exr.nextFloat();
        float C = exr.nextFloat();
        
        byte[] X = Vector.random_byte_vector(length);
        Tensor tX = eg.tensor_int8(X, height, width);
       
        //CPU-------------------------------------------------------------------
//        byte[] Y1 = Vector.img_linear(alpha, X, beta);
        byte[] Y1 = Vector.img_exp(alpha, X, beta, C);
//        byte[] Y1 = Vector.img_log(C, alpha, X, beta);
        
        System.out.print("CPU : "); Vector.println(Y1, 0, 10);
        
        //GPU-------------------------------------------------------------------
//        Tensor tY1 = eg.img.linear(false, alpha, tX, beta).c();
//        Tensor tY2 = eg.img.linear(true, alpha, tX, beta).c();
       
        Tensor tY1 = eg.img.exp(false, alpha, tX, beta, C).c();
        Tensor tY2 = eg.img.exp(true, alpha, tX, beta, C).c();

//        Tensor tY1 = eg.img.log(false, C, alpha, tX, beta).c();
//        Tensor tY2 = eg.img.log(true, C, alpha, tX, beta).c();

        //compare---------------------------------------------------------------
        byte[] Y2 = tY1.pixel();
        System.out.print("GPU1: "); Vector.println(Y2, 0, 10);
        float sp1 = Vector.samePercent_absolute(Y1, Y2); System.out.println("sp1:" + sp1);
        
        byte[] Y3 = tY2.pixel();
        System.out.print("GPU1: "); Vector.println(Y3, 0, 10);
        float sp2 = Vector.samePercent_absolute(Y1, Y3); System.out.println("sp2:" + sp2);
        
        if(sp1 < 0.99) throw new RuntimeException();
        if(sp2 < 0.99) throw new RuntimeException();
        //delete----------------------------------------------------------------
        eg.delete(tX, tY1, tY2);
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
        catch(Exception e)
        {
            e.printStackTrace();
        }
        System.out.println(CudaException.lastException());
    }
}
 