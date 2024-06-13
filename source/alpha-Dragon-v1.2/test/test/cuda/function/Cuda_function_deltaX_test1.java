package test.cuda.function;


import z.dragon.engine.Engine;
import static z.dragon.alpha.Alpha.alpha;
import z.dragon.engine.Tensor;
import z.dragon.engine.cuda.impl.Cuda;
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
public class Cuda_function_deltaX_test1 
{
    static { alpha.home("C:\\Users\\Gilgamesh\\Desktop\\Dragon-alpha-v1.1");}
    static Engine eg = alpha.engine.cuda_float32(0, alpha.engine.memp1());
    static final ExRandom exr = new ExRandom();
    
    public static void testCorrect(int height, int width) {
        int stride = (width + 3) >> 2 << 2;
	int lengthv = height * stride;
        int length = height * width;
        
        System.out.println("\nTest Corret:");
        System.out.format("(height, width, stride) = (%d, %d, %d)\n", height, width, stride);
        System.out.format("(lengthv, length) = (%d, %d)\n", lengthv, length);
        
        float[] Y = Vector.random_float_vector(length, -5, 5);  Vector.println("Y = ", Y, 0, 10);
        Tensor tY = eg.tensor(Y, height, width);
        
        float[] deltaY = Vector.random_float_vector(length, -1, 1); Vector.println("deltaY = ", deltaY, 0, 10);
        Tensor tdeltaY = eg.tensor(deltaY, height, width);
        
        float alpha = exr.nextFloat(0, 0.5f); System.out.println("alpha = " + alpha);
        float beta = exr.nextFloat(0, 0.5f); System.out.println("beta = " + beta);
        float gamma = exr.nextFloat(1, -1);
        float vmin = exr.nextFloat(0, -1.5f);
        float vmax = exr.nextFloat(0, 1.5f);
        float k = exr.nextFloat();
        
        //CPU-------------------------------------------------------------------
        float[] deltaX1 = new float[length];
            
//        float[] deriY = Vector.relu_deri(Y);
//        float[] deriY = Vector.leakyRelu_Deri(Y, k);
        float[] deriY = Vector.gelu_dei(Y);
        
//        float[] deriY = Vector.csc_deri(Y, alpha, beta);
//        float[] deriY = Vector.sec_deri(Y, alpha, beta);

        
        
        Vector.elementMul(deltaY, deriY, deltaX1);//deltaX = deriY * deltaY
        System.out.print("CPU:  "); Vector.println(deltaX1, 0, 10);
        
        //GPU-------------------------------------------------------------------
//        Tensor tdeltaX1 = eg.relu_deltaX_v1(false, tdeltaY, tY).c();
//        Tensor tdeltaX2 = eg.relu_deltaX_v2(true, tdeltaY, tY).c();
        
//        Tensor tdeltaX1 = eg.leakyRelu_deltaX_v1(false, tdeltaY, tY, k).c();
//        Tensor tdeltaX2 = eg.leakyRelu_deltaX_v2(true, tdeltaY, tY, k).c();

        Tensor tdeltaX1 = eg.gelu_deltaX(false, tdeltaY, tY).c();
        Tensor tdeltaX2 = eg.gelu_deltaX(true, tdeltaY, tY).c();
        
        
//        Tensor tdeltaX1 = eg.csc_deltaX(false, tdeltaY, tY, alpha, beta).c();
//        Tensor tdeltaX2 = eg.csc_deltaX(true, tdeltaY, tY, alpha, beta).c();

//        Tensor tdeltaX1 = eg.sec_deltaX(false, tdeltaY, tY, alpha, beta).c();
//        Tensor tdeltaX2 = eg.sec_deltaX(true, tdeltaY, tY, alpha, beta).c();
    
        float[] deltaX2 = eg.valueOf(tdeltaX1);
        System.out.print("GPU1: "); Vector.println(deltaX2, 0, 10);
        
        float[] deltaX3 = eg.valueOf(tdeltaX2);
        System.out.print("GPU2: "); Vector.println(deltaX3, 0, 10);
        
        //compare---------------------------------------------------------------
        float sp1 = Vector.samePercent_relative(deltaX1, deltaX2); System.out.println("sp1:" + sp1);
        float sp2 = Vector.samePercent_relative(deltaX1, deltaX3); System.out.println("sp2:" + sp2);
       
        //delete----------------------------------------------------------------
        if(sp1 < 0.99 || sp2 < 0.99) throw new RuntimeException();
        if(tdeltaX1.hasNan().get()) throw new RuntimeException();
        if(tdeltaX2.hasNan().get()) throw new RuntimeException();
        
        System.gc();
    }
    
    public static void testSpeed(int height, int width)
    {
        int stride = (width + 3) >> 2 << 2;
	int lengthv = height * stride;
        int length = height * width;
        
        System.out.println("\nTest Speed:");
        System.out.format("(height, width, stride) = (%d, %d, %d)\n", height, width, stride);
        System.out.format("(lengthv, length) = (%d, %d)\n", lengthv, length);
        
        float[] Y = Vector.random_float_vector(length);
        float[] deltaY = Vector.random_float_vector(length, -1, 1);

        Tensor tY = eg.tensor(Y, height, width);
        Tensor tdeltaY = eg.tensor(deltaY, height, width);
        
        SimpleTimer timer=new SimpleTimer();
        timer.record();
        int nIter = 1000;
        for(int i=0; i < nIter; i++)
        {
            tdeltaY = eg.sec_deltaX(true, tdeltaY, tY, 1, 1);
//            tdeltaY = eg.csc_deltaX(true, tdeltaY, tY, 1, 1);
        }
        Cuda.deviceSynchronize();
        timer.record();
        
        float time = (float) timer.timeStamp_dif_millis() / nIter;
	int data_size = (lengthv) * 4 * 3;
        
	float speed =  ((float)data_size) / (1 << 30) / (time * 1e-3f);
        System.out.format("Time = %f, Speed = %f GB/s\n", time, speed);
        
        eg.delete(tdeltaY, tY);
    }
    
    public static void main(String[] args)
    {
        Vector.PRINT_DIFFERENT = true;
        
        for(int h=1; h<=10; h++)
            for(int w=1; w<=256; w++) testCorrect(h, w);
        testCorrect(1024, 1024);
        testSpeed(1024, 1024);
    }
}
