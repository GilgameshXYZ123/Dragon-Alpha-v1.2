/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package test.cuda.function;

import static z.dragon.alpha.Alpha.alpha;
import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.dragon.engine.cuda.impl.Cuda;
import z.util.lang.SimpleTimer;
import z.util.math.ExRandom;
import z.util.math.vector.Vector;

/**
 *
 * @author Gilgamesh
 */
public class Cuda_function_fusion_test1
{
    static { alpha.home("C:\\Users\\Gilgamesh\\Desktop\\Dragon-alpha-v1.2");}
    static Engine eg = alpha.engine.cuda_float32(0, alpha.engine.memp1());
    static final ExRandom exr = new ExRandom();
    
    public static void testCorrect(int height, int width) {
        int stride = (width + 3) >> 2 << 2;
	int lengthv = height * stride;
        int length = height * width;
        
        System.out.println("\nTest Corret:");
        System.out.format("(height, width, stride) = (%d, %d, %d)\n", height, width, stride);
        System.out.format("(lengthv, length) = (%d, %d)\n", lengthv, length);
        
        float[] X1 = Vector.random_float_vector(length, -1f, 1f); 
        float[] X2 = Vector.random_float_vector(length, -1f, 1f); 
        Tensor tX1 = eg.tensor(X1, height, width);
        Tensor tX2 = eg.tensor(X2, height, width);
        
        float alpha = exr.nextFloat(0, 0.5f), alpha2 = exr.nextFloat();
        float beta = exr.nextFloat(1, -1), beta2 = exr.nextFloat();
        float gamma = exr.nextFloat();
        float k = exr.nextFloat();
        
        //CPU-------------------------------------------------------------------
        float[] Y1 = Vector.linear2(X1, X2, alpha, beta, gamma);
        
//        Y1 = Vector.leakyRelu(Y1, k);
        Y1 = Vector.elu(Y1, alpha2, k);
//        Y1 = Vector.softplus(Y1);
//        Y1 = Vector.gelu(Y1);
//        Y1 = Vector.sigmoid(Y1);
//        Y1 = Vector.tanh(Y1);
        
        System.out.print("CPU : "); Vector.println(Y1, 0, 10);
        
        //GPU-------------------------------------------------------------------
//        Tensor tY1 = eg.linear2_leakyRelu(false, tX1, tX2, alpha, beta, gamma, k);
//        Tensor tY2 = eg.linear2_leakyRelu(true , tX1, tX2, alpha, beta, gamma, k);

        Tensor tY1 = eg.linear2_elu(false, tX1, tX2, alpha, beta, gamma, alpha2, k);
        Tensor tY2 = eg.linear2_elu(true,  tX1, tX2, alpha, beta, gamma, alpha2, k);
            
//        Tensor tY1 = eg.linear2_softplus(false, tX1, tX2, alpha, beta, gamma);
//        Tensor tY2 = eg.linear2_softplus(true,  tX1, tX2, alpha, beta, gamma);
        
//        Tensor tY1 = eg.linear2_gelu(false, tX1, tX2, alpha, beta, gamma);
//        Tensor tY2 = eg.linear2_gelu(true,  tX1, tX2, alpha, beta, gamma);

//        Tensor tY1 = eg.linear2_sigmoid(false, tX1, tX2, alpha, beta, gamma);
//        Tensor tY2 = eg.linear2_sigmoid(true,  tX1, tX2, alpha, beta, gamma);

//        Tensor tY1 = eg.linear2_tanh(false, tX1, tX2, alpha, beta, gamma);
//        Tensor tY2 = eg.linear2_tanh(true,  tX1, tX2, alpha, beta, gamma);

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
        int stride = (width + 3) >> 2 << 2;
	int lengthv = height * stride;
        int length = height * width;
        
        System.out.println("\nTest Speed:");
        System.out.format("(height, width, stride) = (%d, %d, %d)\n", height, width, stride);
        System.out.format("(lengthv, length) = (%d, %d)\n", lengthv, length);
        
        float[] X1 = Vector.random_float_vector(length);
        float[] X2 = Vector.random_float_vector(length, -1, 1);

        Tensor tX1 = eg.tensor(X1, height, width);
        Tensor tX2 = eg.tensor(X2, height, width);
        
        SimpleTimer timer = new SimpleTimer().record();
        int nIter = 1000;
        for(int i=0; i < nIter; i++) {
            Tensor tY = eg.linear2_elu(false, tX1, tX2, 1, 1, 1, width, 1);
//            Tensor tY = eg.linear2_softplus(false, tX1, tX2, 1, 2, 3).c();
//            Tensor tY = eg.linear2_gelu(false, tX1, tX2, 1, 2, 3).c();
//            Tensor tY = eg.linear2_sigmoid(false, tX1, tX2, 1, 1, 0);
//            Tensor tY = eg.linear2_tanh(false, tX1, tX2, 1, 1, 0);
            tY.delete();
        }
        Cuda.deviceSynchronize();
        timer.record();
        
        float time = (float) timer.timeStamp_dif_millis() / nIter;
	int data_size = (lengthv) * 4 * 3;
        
	float speed =  ((float)data_size) / (1 << 30) / (time * 1e-3f);
        System.out.format("Time = %f, Speed = %f GB/s\n", time, speed);
        
        eg.delete(tX2, tX1);
    }
    
    public static void main(String[] args) {
        Vector.PRINT_DIFFERENT = true;
        for(int h=1; h<=10; h++)
            for(int w=1; w<=256; w++) testCorrect(h, w);
        testCorrect(1024, 1024);
        
        testSpeed(2048, 2048);
    }
}
