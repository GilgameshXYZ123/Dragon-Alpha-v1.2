/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package test.cuda.batchnorm;

import static z.dragon.alpha.Alpha.alpha;
import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.util.math.ExRandom;
import z.util.math.vector.Vector;

/**
 *
 * @author Gilgamesh
 */
public class Fused_Affine_LeakyRelu_backward1 
{
    static { alpha.home("C:\\Users\\Gilgamesh\\Desktop\\Dragon-alpha-v1.2");}
    static Engine eg = alpha.engine.cuda_float32(0, alpha.engine.memp1());
    static final ExRandom exr = new ExRandom();
    
    //affine = false
    public static void testCorrect(int height, int width) {
        System.out.format("\ntestCorrect: (height, width) = (%d, %d)\n", height, width);
        
        float eps = exr.nextFloat();
        float k = exr.nextFloat();
        System.out.format("(eps, k) = (%f, %f)\n", eps, k);
        
        Tensor A = eg.Gaussian(width);
        Tensor B = eg.Gaussian(width);
        
        Tensor deltaY = eg.Uniform(height, width);
        Tensor X = eg.uniform(-1, 1, height, width);
       
        float alpha = exr.nextFloat();
        
        //path1-----------------------------------------------------------------
        Tensor tY1 = eg.affine(false, X, A, B);
        
//        Tensor tY2 = eg.leakyRelu(false, tY1, k);
//        Tensor tY2 = eg.elu(false, tY1, alpha, k);
//        Tensor tY2 = eg.softplus(false, tY1);
        Tensor tY2 = eg.gelu(false, tY1);
//        Tensor tY2 = eg.sigmoid(false, tY1);
//        Tensor tY2 = eg.tanh(false, tY1);
        
//        Vector.println("X : ", X.value(), 0, 10);
//        Vector.println("Y1: ", tY1.value(), 0, 10);
//        Vector.println("Y2: ", tY2.value(), 0, 10);
    
//        Tensor tdeltaY1 = eg.leakyRelu_deltaX_v1(false, deltaY, tY2, k);
//        Tensor tdeltaY1 = eg.elu_deltaX_v1(false, deltaY, tY2, alpha, k);
//        Tensor tdeltaY1 = eg.softplus_deltaX_v1(false, deltaY, tY2);
        Tensor tdeltaY1 = eg.gelu_deltaX(false, deltaY, tY1);
//        Tensor tdeltaY1 = eg.sigmoid_deltaX_v1(false, deltaY, tY2);
//        Tensor tdeltaY1 = eg.tanh_deltaX_v1(false, deltaY, tY2);

        Tensor[] grads1 = eg.affine_deltaAB_v1(tdeltaY1, tY1, A, B);
        Tensor tdeltaX1 = eg.mul_row(true, tdeltaY1, A);
        Tensor tdeltaA1 = grads1[0];
        Tensor tdeltaB1 = grads1[1];
        
        //path2-----------------------------------------------------------------
//        Tensor tY3 = eg.affine_leakyRelu(true, X, A, B, k);
//        Tensor tY3 = eg.affine_elu(true, X, A, B, alpha, k);
        Tensor tY3 = eg.affine_gelu(false, X, A, B);
//        Tensor tY3 = eg.affine_softplus(true, X, A, B);
//        Tensor tY3 = eg.affine_sigmoid(true, X, A, B);
//        Tensor tY3 = eg.affine_tanh(true, X, A, B);
        
//        Tensor[] grads2 = eg.affine_leakyRelu_deltaAB_v1(deltaY, k, tY3, A, B);
//        Tensor[] grads2 = eg.affine_elu_deltaAB_v1(deltaY, alpha, k, tY3, A, B);
//        Tensor[] grads2 = eg.affine_softplus_deltaAB_v1(deltaY, tY3, A, B);
        Tensor[] grads2 = eg.affine_gelu_deltaAB_v2(deltaY, X, A, B);
//        Tensor[] grads2 = eg.affine_sigmoid_deltaAB_v1(deltaY, tY3, A, B);
//        Tensor[] grads2 = eg.affine_tanh_deltaAB_v1(deltaY, tY3, A, B);
                
//        Tensor tdeltaX2 = eg.affine_leakyRelu_deltaX_v1(false, deltaY, k, tY3, A);
//        Tensor tdeltaX2 = eg.affine_elu_deltaX_v1(false, deltaY, alpha, k, tY3, A);
        Tensor tdeltaX2 = eg.affine_gelu_deltaX_v2(false, deltaY, X, A, B);
//        Tensor tdeltaX2 = eg.affine_softplus_deltaX_v1(false, deltaY, tY3, A);
//        Tensor tdeltaX2 = eg.affine_sigmoid_deltaX_v1(false, deltaY, tY3, A);
//        Tensor tdeltaX2 = eg.affine_tanh_deltaX_v1(false, deltaY, tY3, A);
        
        Tensor tdeltaA2 = grads2[0];
        Tensor tdeltaB2 = grads2[1];
        
        //compare---------------------------------------------------------------
        float[] Y1 = tY2.value(), Y2 = tY3.value();
        float[] dX1 = tdeltaX1.value(), dX2 = tdeltaX2.value();
        float[] dA1 = tdeltaA1.value(), dA2 = tdeltaA2.value();
        float[] dB1 = tdeltaB1.value(), dB2 = tdeltaB2.value();
        
        float sp0 = Vector.samePercent_relative(Y1, Y2);
        float sp1 = Vector.samePercent_relative(dX1, dX2, 0.005f);
        float sp2 = Vector.samePercent_relative(dA1, dA2);
        float sp3 = Vector.samePercent_relative(dB1, dB2);
        
        Vector.println("Y1 = ", Y1, 0, 10);
        Vector.println("Y2 = ", Y2, 0, 10);
        System.out.println("sp0 (Y) = " + sp0);
        
        Vector.println("dX1 = ", dX1, 0, 10);
        Vector.println("dX2 = ", dX2, 0, 10);
        System.out.println("sp1 (deltaX) = " + sp1);
 
        Vector.println("dA1 = ", dA1, 0, 10);
        Vector.println("dA2 = ", dA2, 0, 10);
        System.out.println("sp2 (deltaA) = " + sp2);
        
        Vector.println("dB1 = ", dB1, 0, 10);
        Vector.println("dB2 = ", dB2, 0, 10);
        System.out.println("sp3 (deltaB) = " + sp3);
        
        if(sp0 < 0.95f) throw new RuntimeException();
        if(sp1 < 0.95f) throw new RuntimeException();
        if(sp2 < 0.95f) throw new RuntimeException();
        if(sp3 < 0.95f) throw new RuntimeException();
        System.gc();
    }
    
    
    public static void main(String[] args)
    {
        Vector.PRINT_DIFFERENT = true;
        
//        for(int h=2; h<=32; h++)
//            for(int w=1; w<=256; w++) testCorrect(h, w);
            
        for(int h=100; h<=105; h++)
            for(int w= 128; w<=256; w++) testCorrect(h, w);
            
        for(int h=1000; h<=1028; h++)
            for(int w= 233; w<=256; w++) testCorrect(h, w);
    }
}
