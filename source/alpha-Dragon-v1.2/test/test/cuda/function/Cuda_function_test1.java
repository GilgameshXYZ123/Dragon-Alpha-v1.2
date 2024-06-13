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
public class Cuda_function_test1 
{
    static { alpha.home("C:\\Users\\Gilgamesh\\Desktop\\Dragon-alpha-v1.2");}
    static Engine eg = alpha.engine.cuda_float32(0, alpha.engine.memp1()).sync(false);
    static final ExRandom exr = new ExRandom();
    
    public static void testCorrect(int height, int width) {
        int stride = (width + 3) >> 2 << 2;
	int lengthv = height * stride;
        int length = height * width;
        
        System.out.println("\nTest Corret:");
        System.out.format("(height, width, stride) = (%d, %d, %d)\n", height, width, stride);
        System.out.format("(lengthv, length) = (%d, %d)\n", lengthv, length);
        
        float[] X1 = Vector.random_float_vector(length, -1f, 1f); Tensor tX1 = eg.tensor(X1, height, width);
        float[] X2 = Vector.random_float_vector(length, -1f, 1f); Tensor tX2 = eg.tensor(X2, height, width);
        
        float[] deltaY = Vector.random_float_vector(length, -1, 1);
        
        float alpha = exr.nextFloat(0, 0.5f), alpha2 = exr.nextFloat();
        float beta = exr.nextFloat(1, -1), beta2 = exr.nextFloat();
        float gamma = exr.nextFloat();
        
        float vmin = exr.nextFloat(0, -1.5f);
        float vmax = exr.nextFloat(0, 1.5f);
        float k = exr.nextFloat();
        
        //CPU-------------------------------------------------------------------
//        float[] Y1 = Vector.linear_greater_switch(alpha, X2, beta, vmin, vmax);
//        float[] Y1 = Vector.linear_greater_switch_mul(alpha, X1, beta, X2, vmin, vmax);
        float[] Y1 = Vector.linear_bound_switch_mul(alpha, X1, vmin, vmax, X2, alpha, beta, gamma);
        
        //float[] Y1 = Vector.leakyRelu(X1, k);
//        float[] Y1 = Vector.gelu(X);
        
//        float[] Y1 = Vector.quadratic2(X1, X2, alpha, beta, alpha2, beta2, gamma, k);
        
//        float[] Y1 = Vector.csc(X, alpha, beta);
//        float[] Y1 = Vector.sec(X, alpha, beta);

//        float[][] mX = Vector.to2D(X, height, width);
//        float[][] mY = new float[height][width];
//        Matrix.softmax(mX, mY, height, width);
//        float[] Y1 = Vector.flatten(mY);
        
        System.out.print("CPU : "); Vector.println(Y1, 0, 10);
        
        //GPU-------------------------------------------------------------------
//        Tensor tY1 = eg.linear_greater_switch(false, alpha, tX2, beta, vmin, vmax).c();
//        Tensor tY2 = eg.linear_greater_switch(true, alpha, tX2, beta, vmin, vmax).c();
        
//        Tensor tY1 = eg.linear_greater_switch_mul(false, alpha, tX1, beta, tX2, vmin, vmax).c();
//        Tensor tY2 = eg.linear_greater_switch_mul(true, alpha, tX1, beta, tX2, vmin, vmax).c();

        Tensor tY1 = eg.linear_bound_switch_mul(false, alpha, tX1, vmin, vmax, tX2, alpha, beta, gamma).c();
        Tensor tY2 = eg.linear_bound_switch_mul(true, alpha, tX1, vmin, vmax, tX2, alpha, beta, gamma).c();
        
//        Tensor tY1 = eg.quadratic2(false, tX1, tX2, alpha, beta, alpha2, beta2, gamma, k);
//        Tensor tY2 = eg.quadratic2(false, tX1, tX2, alpha, beta, alpha2, beta2, gamma, k);
         
//        Tensor tY1 = eg.softmax(false, tX).c();
//        Tensor tY2 = eg.softmax(true, tX).c();
        
//        Tensor tY1 = eg.softmax_crossEntropy(tX, tX2);
//        Tensor tY2 = eg.softmax_crossEntropy(tX, tX2);
                
//        Tensor tY1 = eg.leakyRelu(false, tX1, k).c();
//        Tensor tY2 = eg.leakyRelu(true, tX1, k).c();

//        Tensor tY1 = eg.gelu(false, tX1);
//        Tensor tY2 = eg.gelu(true, tX1);

//        Tensor tY1 = eg.csc(false, alpha, tX, beta).c();
//        Tensor tY2 = eg.csc(true, alpha, tX, beta).c();
        
//        Tensor tY1 = eg.sec(false, alpha, tX, beta).c();
//        Tensor tY2 = eg.sec(true, alpha, tX, beta).c();

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
        eg.delete(tX1, tX2, tY1);
    }
    
    public static void testSpeed(int height, int width) {
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
            Tensor tY = eg.leakyRelu(true, tX2);
//            Tensor tY = eg.csc(true, 1, tX1, 1);
//            Tensor tY = eg.sec(true, 1, tX2, 1);
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
            SimpleTimer timer = SimpleTimer.clock();
            
//            Vector.PRINT_DIFFERENT = true;
//            
//            for(int h = 1; h <= 20; h++)
//                for(int w = 1; w <= 256; w++) testCorrect(h, w);
////            
//            for(int h=100; h<=105; h++)
//                for(int w= 40; w<=64; w++) testCorrect(h, w);
//            
//            for(int h=300; h<=305; h++)
//                for(int w=7; w<=12; w++) testCorrect(h, w);
//            
//            for(int h=300; h<=305; h++)
//                for(int w= 450; w<=512; w++) testCorrect(h, w);
//            
//            testCorrect(512, 8191);
//            testCorrect(511, 8192);
//            testCorrect(512, 8192);
//            testCorrect(1024, 8192);
//            
//            System.out.println(timer.record().timeStamp_dif_millis());//5877, 5567
            
            testSpeed(64, 32*32*128);
//            testSpeed(512, 4*4*512);
        }
        catch(Exception e)
        {
            e.printStackTrace();
        }
        System.out.println(CudaException.lastException());
    }
}
 