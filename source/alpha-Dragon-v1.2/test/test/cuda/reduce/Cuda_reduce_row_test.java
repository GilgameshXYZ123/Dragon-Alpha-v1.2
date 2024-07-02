package test.cuda.reduce;


import z.dragon.engine.Engine;
import static z.dragon.alpha.Alpha.alpha;
import z.dragon.engine.Tensor;
import z.dragon.engine.cuda.impl.Cuda;
import z.dragon.engine.cuda.impl.CudaException;
import z.util.lang.SimpleTimer;
import z.util.math.ExRandom;
import z.util.math.vector.Matrix;
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
public class Cuda_reduce_row_test 
{
    static { alpha.home("C:\\Users\\Gilgamesh\\Desktop\\Dragon-alpha-v1.1");}
    static Engine eg = alpha.engine.cuda_float32(0, alpha.engine.memp1());
    static final ExRandom exr = new ExRandom();
    
    public static void testCorrect(int height, int width) {
        int stride = (width + 3) >> 2 << 2;
	int lengthv = height * 3 * 2 * stride;
        int length  = height * 3 * 2 * width;
        
        System.out.println("\nTest Corret:");
        System.out.format("(height, width, stride) = (%d, %d, %d)\n", height, width, stride);
        System.out.format("(lengthv, length) = (%d, %d)\n", lengthv, length);
        
        float[] X = Vector.random_float_vector(length, 3f, 6f);
        float[] X2 = Vector.random_float_vector(length, 0, 1);
        Tensor tX = eg.tensor(X, height, 3, 2, width);
        Tensor tX2 = eg.tensor(X2, height, 3, 2, width);
        
        float alpha = exr.nextFloat(-1f, 1f), alpha2 = exr.nextFloat(-1f, 1f);
        float beta  = exr.nextFloat(-1f, 1f), beta2  = exr.nextFloat(-1f, 1f);
        float gamma = exr.nextFloat(-1f, 1f), gamma2 = exr.nextFloat(-1f, 1f);
        
        //CPU-------------------------------------------------------------------
        float[][] mX = Matrix.toMatrix(X, 3 * 2 * width);
        float[][] mX2 = Matrix.toMatrix(X2, 3 * 2 * width);
        float[] Y1 = new float[height];
        
//        Y1 = Matrix.row_mean(mX);
//        Y1 = Matrix.row_squareMean(mX);
        Y1 = Matrix.row_linear(mX, alpha, beta);
//        Y1 = Matrix.row_linear2(mX, mX2, alpha, beta, gamma);
//        Matrix.row_quadratic(mX, alpha, beta, gamma, Y1);
//        Y1 = Matrix.row_quadratic2(mX, mX2, alpha, beta, gamma, alpha2, beta2, gamma2);
        
//        Matrix.maxValueEachRow(mX, Y1);
//        Matrix.minValue_each_row(mX, Y1);
     
        System.out.print("CPU: "); Vector.println(Y1, 0, 10);
        //GPU-------------------------------------------------------------------
//        Tensor tY = eg.row_mean(tX, 3 * 2 * width).c();
//        Tensor tY = eg.row_linear(tX, 3 * 2 * width, alpha, beta).c();
//        Tensor tY = eg.row_linear2(tX, tX2, 3 * 2 * width, alpha, beta, gamma);
//        Tensor tY = eg.row_squareMean(tX, 3 * 2 * width).c();
        Tensor tY = eg.row_linear_quadratic(tX, 2 * 3 * width, alpha, beta, alpha, beta, gamma)[0].c();
//        Tensor tY = eg.row_quadratic(tX, 3 * 2 * width, alpha, beta, gamma).c();
//        Tensor tY = eg.row_quadratic2(tX, tX2, 3 * 2 * width, alpha, beta, gamma, alpha2, beta2, gamma2);
//        Tensor tY = eg.reduce_row_max(tX, 3 * 2 * width).c();
//        Tensor tY = eg.reduce_row_min(tX, 3 * 2 * width).c();
    
        float[] Y2 = eg.valueOf(tY); 
        System.out.print("tY.dim:"); Vector.println(tY.dim());
        System.out.print("GPU: "); Vector.println(Y2, 0, 10);
        //compare---------------------------------------------------------------
        
        float sp0 = Vector.samePercent_relative(Y1, Y2, 2e-4f); System.out.println("sp0:" + sp0);
        if(sp0 < 0.95) throw new RuntimeException();
      
        eg.delete(tY, tX);
    }
    
    public static void testSpeed(int height, int width) {
        eg.check(false);
        eg.sync(false);
        int stride = (width + 3) >> 2 << 2;
	int lengthv = height * stride;
        int length  = height * width;
        
        System.out.println("\nTest Corret:");
        System.out.format("(height, width, stride) = (%d, %d, %d)\n", height, width, stride);
        System.out.format("(lengthv, length) = (%d, %d)\n", lengthv, length);
        
        float[] X = Vector.random_float_vector(length, -2f, -1f);
        Tensor tX = eg.tensor(X, height, width);
        
        SimpleTimer timer = new SimpleTimer();
        timer.record();
        int nIter = 1000;
        for(int i=0; i < nIter; i++)
        {
//            Tensor tY = eg.reduce_row_quadratic2Sum(tX, width, 1, 2, 3).c(); eg.delete(tY);
//            Tensor tY = eg.reduce_row_max(tX, width).c(); eg.delete(tY);
            Tensor tY = eg.row_min(tX, width).c(); eg.delete(tY);
//            Tensor tY = eg.tensor(height, width).c(); eg.delete(tY);
        }
        Cuda.deviceSynchronize();
        timer.record();
        float time = (float) timer.timeStamp_dif_millis()/nIter;
	int data_size = (lengthv) * 4 * 1;
	float speed =  ((float)data_size) / (1 << 30) / (time * 1e-3f);
        System.out.format("Time = %f, Speed = %f GB/s\n", time, speed);
        eg.delete(tX);
    }
    public static void main(String[] args)
    {
        try
        {
            Vector.PRINT_DIFFERENT = true;
//            (3, 1), (3, 2,
            for(int h = 1; h <= 20; h++)
                for(int w = 1; w <= 256; w++) testCorrect(h, w);
//            
            for(int h=100; h<=105; h++)
                for(int w= 40; w<=64; w++) testCorrect(h, w);
            
            for(int h=300; h<=305; h++)
                for(int w=7; w<=12; w++) testCorrect(h, w);
            
            for(int h=300; h<=305; h++)
                for(int w= 450; w<=512; w++) testCorrect(h, w);
            
//            testCorrect(1, 85);
//            
            testCorrect(1024, 1024);
            testSpeed(1024, 1024);
//            testCorrect(1, 1024*1024);
//            testSpeed(1, 1024*1024);
            
        }
        catch(Exception e)
        {
            e.printStackTrace();
        }
        System.out.println(CudaException.lastException());
    }
}
