package test.cuda.function;


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
public class Cuda_function_test3 
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
        
        float[] X1 = Vector.random_float_vector(length, -1f, 1f); 
        float[] X2 = Vector.random_float_vector(length, -1f, 1f); 
        float[] X3 = Vector.random_float_vector(length, -1f, 1f);
        
        Tensor tX1 = eg.tensor(X1, height, width);
        Tensor tX2 = eg.tensor(X2, height, width);
        Tensor tX3 = eg.tensor(X3, height, width);
        
        float alpha1 = exr.nextFloat(0, 0.5f), beta1 = exr.nextFloat(1, -1);
        float alpha2 = exr.nextFloat(0, 0.5f), beta2 = exr.nextFloat(1, -1);
        float alpha3 = exr.nextFloat(0, 0.5f), beta3 = exr.nextFloat(1, -1);
        float gamma = exr.nextFloat();
        
        float vmin = exr.nextFloat(0, -1.5f);
        float vmax = exr.nextFloat(0, 1.5f);
        float k = exr.nextFloat();
        
        //CPU-------------------------------------------------------------------
//        float[] Y1 = Vector.mul_linear2(X1, X2, X3, alpha3, beta3, gamma);
        float[] Y1 = Vector.mul_squareDiv(alpha1, X1, beta1, alpha2, X2, beta2, alpha3, X3, beta3, gamma);
        System.out.print("CPU : "); Vector.println(Y1, 0, 10);
        
        
        //GPU-------------------------------------------------------------------
//        Tensor tY1 = eg.mul_linear2(false, tX1, tX2, tX3, alpha3, beta3, gamma);
//        Tensor tY2 = eg.mul_linear2(true,  tX1, tX2, tX3, alpha3, beta3, gamma);
           
        Tensor tY1 = eg.mul_squareDiv(false, alpha1, tX1, beta1, alpha2, tX2, beta2, alpha3, tX3, beta3, gamma);
        Tensor tY2 = eg.mul_squareDiv(true, alpha1, tX1, beta1, alpha2, tX2, beta2, alpha3, tX3, beta3, gamma);
        
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
        eg.delete(tX1, tX2, tX3, tY1);
    }
    
    public static void testSpeed(int height, int width)
    {
        eg.check(false);
        int stride = (width + 3) >> 2 << 2;
	int lengthv = height * stride;
        int length = height * width;
        
        System.out.println("\nTest Speed:");
        System.out.format("(height, width, stride) = (%d, %d, %d)\n", height, width, stride);
        System.out.format("(lengthv, length) = (%d, %d)\n", lengthv, length);
        
        float[] X1 = Vector.random_float_vector(length);
        float[] X2 = Vector.random_float_vector(length);
        Tensor tX1 = eg.tensor(X1, height, width);
        Tensor tX2 = eg.tensor(X1, 1, height, width);
        
        SimpleTimer timer = new SimpleTimer();
        timer.record();
        int nIter = 1000;
        for(int i=0; i < nIter; i++)
        {
//            Tensor tY = eg.csc(true, 1, tX1, 1);
            Tensor tY = eg.sec(true, 1, tX2, 1);
        }
        Cuda.deviceSynchronize();
        timer.record();
        float time = (float) timer.timeStamp_dif_millis()/nIter;
	int data_size = (lengthv) * 4 * 2;
	float speed =  ((float)data_size) / (1 << 30) / (time * 1e-3f);
        System.out.format("Time = %f, Speed = %f GB/s\n", time, speed);
        eg.delete(tX1);
    }
    
    public static void main(String[] args)
    {
        try
        {
            Vector.PRINT_DIFFERENT = true;
            
            for(int h=1; h<=10; h++)
                for(int w=1; w<=256; w++) testCorrect(h, w);
        
            testCorrect(1024, 1024);
            testSpeed(1024, 1024);
        }
        catch(Exception e)
        {
            e.printStackTrace();
        }
        System.out.println(CudaException.lastException());
    }
}
 