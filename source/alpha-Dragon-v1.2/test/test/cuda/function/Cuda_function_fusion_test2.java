/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package test.cuda.function;

import static z.dragon.alpha.Alpha.alpha;
import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.util.math.ExRandom;
import z.util.math.vector.Vector;

/**
 *
 * @author Gilgamesh
 */
public class Cuda_function_fusion_test2
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
        float[] deltaY = Vector.random_float_vector(length, -1f, 1f);
        Tensor tX1 = eg.tensor(X1, height, width);
        Tensor tX2 = eg.tensor(X2, height, width);
        Tensor tdeltaY = eg.tensor(deltaY, height, width);
        
        float alpha = exr.nextFloat(0, 0.5f), alpha2 = exr.nextFloat();
        float beta = exr.nextFloat(1, -1), beta2 = exr.nextFloat();
        float gamma = exr.nextFloat();
        float k = exr.nextFloat();
        
        //GPU-------------------------------------------------------------------
//        Tensor tY1 = eg.linear2_leakyRelu(false, tX1, tX2, alpha, beta, gamma, k);
//        Tensor[] grads1 = eg.linear2_leakyRelu_deltaX_v1(false, tdeltaY, tY1, alpha, beta, k);
//        Tensor[] grads2 = eg.linear2_leakyRelu_deltaX_v2(false, tdeltaY, tX1, tX2, alpha, beta, gamma, k);
//        Tensor[] grads3 = eg.linear_2out(false, eg.leakyRelu_deltaX_v1(false, tdeltaY, tY1, k), alpha, 0, beta, 0);
        
//        Tensor tY1 = eg.linear2_elu(false, tX1, tX2, alpha, beta, gamma, alpha2, k);
//        Tensor[] grads1 = eg.linear2_elu_deltaX_v1(false, tdeltaY, tY1, alpha, beta, alpha2, k);
//        Tensor[] grads2 = eg.linear2_elu_deltaX_v2(false, tdeltaY, tX1, tX2, alpha, beta, gamma,  alpha2, k);
//        Tensor[] grads3 = eg.linear_2out(false, eg.elu_deltaX_v1(false, tdeltaY, tY1, alpha2, k), alpha, 0, beta, 0);

//        Tensor tY1 = eg.linear2_softplus(false, tX1, tX2, alpha, beta, gamma);
//        Tensor[] grads1 = eg.linear2_softplus_deltaX_v1(false, tdeltaY, tY1, alpha, beta);
//        Tensor[] grads2 = eg.linear2_softplus_deltaX_v2(false, tdeltaY, tX1, tX2, alpha, beta, gamma);
//        Tensor[] grads3 = eg.linear_2out(false, eg.softplus_deltaX_v1(false, tdeltaY, tY1), alpha, 0, beta, 0);

        Tensor tY1 = eg.linear2(false, tX1, tX2, alpha, beta, gamma);
        Tensor[] grads1 = eg.linear2_gelu_deltaX_v2(false, tdeltaY, tX1, tX2, alpha, beta, gamma);
        Tensor[] grads2 = eg.linear2_gelu_deltaX_v2(false, tdeltaY, tX1, tX2, alpha, beta, gamma);
        Tensor[] grads3 = eg.linear_2out(false, eg.gelu_deltaX(false, tdeltaY, tY1), alpha, 0, beta, 0);

//        Tensor tY1 = eg.linear2_sigmoid(false, tX1, tX2, alpha, beta, gamma);
//        Tensor[] grads1 = eg.linear2_sigmoid_deltaX_v1(false, tdeltaY, tY1, alpha, beta);
//        Tensor[] grads2 = eg.linear2_sigmoid_deltaX_v2(false, tdeltaY, tX1, tX2, alpha, beta, gamma);
//        Tensor[] grads3 = eg.linear_2out(false, eg.sigmoid_deltaX_v1(false, tdeltaY, tY1), alpha, 0, beta, 0);

//        Tensor tY1 = eg.linear2_tanh(false, tX1, tX2, alpha, beta, gamma);
//        Tensor[] grads1 = eg.linear2_tanh_deltaX_v1(false, tdeltaY, tY1, alpha, beta);
//        Tensor[] grads2 = eg.linear2_tanh_deltaX_v2(false, tdeltaY, tX1, tX2, alpha, beta, gamma);
//        Tensor[] grads3 = eg.linear_2out(false, eg.tanh_deltaX_v1(false, tdeltaY, tY1), alpha, 0, beta, 0);
        
        Tensor dA1 = grads1[0], dB1 = grads1[1];
        Tensor dA2 = grads2[0], dB2 = grads2[1];
        Tensor dA3 = grads3[0], dB3 = grads3[1];

        //compare---------------------------------------------------------------
        float[] A1 = eg.valueOf(dA1), B1 = eg.valueOf(dB1);
        float[] A2 = eg.valueOf(dA2), B2 = eg.valueOf(dB2);
        float[] A3 = eg.valueOf(dA3), B3 = eg.valueOf(dB3);
        
        Vector.println("A1 = ", A1,  0, 10);
        Vector.println("A2 = ", A2,  0, 10);
        Vector.println("A3 = ", A3,  0, 10);
        
        Vector.println("B1 = ", A1,  0, 10);
        Vector.println("B2 = ", A2,  0, 10);
        Vector.println("B3 = ", A3,  0, 10);
        
        float spA1 = Vector.samePercent_relative(A1, A3); 
        float spA2 = Vector.samePercent_relative(A2, A3); 
        float spB1 = Vector.samePercent_relative(B1, B3); 
        float spB2 = Vector.samePercent_relative(B1, B3); 
        System.out.println("spA1:" + spA1);
        System.out.println("spA2:" + spA2);
        System.out.println("spB1:" + spB1);
        System.out.println("spB2:" + spB2);
        
        if(spA1 < 0.99) throw new RuntimeException();
        if(spA2 < 0.99) throw new RuntimeException();
        if(spB1 < 0.99) throw new RuntimeException();
        if(spB2 < 0.99) throw new RuntimeException();
        
        if(spA1 < 0.99) throw new RuntimeException();
        if(spA2 < 0.99) throw new RuntimeException();
        if(spB1 < 0.99) throw new RuntimeException();
        if(spB2 < 0.99) throw new RuntimeException();
        System.gc();
    }
    
    public static void main(String[] args) {
        Vector.PRINT_DIFFERENT = true;
        
        for(int h=1; h<=10; h++)
            for(int w=1; w<=256; w++) testCorrect(h, w);
        testCorrect(1024, 1024);
    }
}
