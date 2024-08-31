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
public class Fused_BatchNorm_LeakyRelu_backward4 
{
    static { alpha.home("C:\\Users\\Gilgamesh\\Desktop\\Dragon-alpha-v1.2");}
    static Engine eg = alpha.engine.cuda_float32(0, alpha.engine.memp1());
    static final ExRandom exr = new ExRandom();
    
    //affine = false
    public static void testCorrect(int height, int width) {
        System.out.format("\ntestCorrect: (height, width) = (%d, %d)\n", height, width);
        
        float eps = exr.nextFloat();
        float alpha = exr.nextFloat();
        float k = exr.nextFloat();
        System.out.format("(eps, k) = (%f, %f)\n", eps, k);
        
        Tensor A = eg.Gaussian(width);
        Tensor B = eg.Gaussian(width);
        
        Tensor deltaY = eg.Uniform(height, width);
        Tensor X = eg.uniform(0, 1, height, width);
        Tensor[] stats = eg.field_var_mean(false, X);
        Tensor X_var = stats[0];
        Tensor X_mean = stats[1];
       
        //path1-----------------------------------------------------------------
        Tensor tY1 = eg.batchNorm(false, X, X_mean, X_var, eps);
        
//        Tensor tY2 = eg.leakyRelu(false, tY1, k);
//        Tensor tY2 = eg.elu(false, tY1, alpha, k);
//        Tensor tY2 = eg.softplus(false, tY1);
        Tensor tY2 = eg.gelu(false, tY1);
//        Tensor tY2 = eg.sigmoid(false, tY1);
//        Tensor tY2 = eg.tanh(false, tY1);
        
//        Vector.println("X : ", X.value(), 0, 10);
//        Vector.println("Y1: ", tY1.value(), 0, 10);
//        Vector.println("Y2: ", tY2.value(), 0, 10);

//        Tensor tdeltaX1 = eg.batchNorm_deltaX_v1(false,
//                eg.leakyRelu_deltaX_v1(false, deltaY, tY2, k), 
//                tY1, X_var, eps);
//        Tensor tdeltaX1 = eg.batchNorm_deltaX_v1(false,
//                eg.elu_deltaX_v1(false, deltaY, tY2, alpha, k), 
//                tY1, X_var, eps);
        Tensor tdeltaX1 = eg.batchNorm_deltaX_v1(false,
                eg.gelu_deltaX(false, deltaY, tY1), 
                tY1, X_var, eps);
//        Tensor tdeltaX1 = eg.batchNorm_deltaX_v1(false,
//                eg.softplus_deltaX_v1(false, deltaY, tY2), 
//                tY1, X_var, eps);
//        Tensor tdeltaX1 = eg.batchNorm_deltaX_v1(false,
//                eg.sigmoid_deltaX_v1(false, deltaY, tY2), 
//                tY1, X_var, eps);
//        Tensor tdeltaX1 = eg.batchNorm_deltaX_v1(false,
//                eg.tanh_deltaX_v1(false, deltaY, tY2), 
//                tY1, X_var, eps);
        
        //path2-----------------------------------------------------------------
//        Tensor tY3 = eg.batchNorm_leakyRelu(false, X, X_mean, X_var, eps, k);
//        Tensor tY3 = eg.batchNorm_elu(false, X, X_mean, X_var, eps, alpha, k);
//        Tensor tY3 = eg.batchNorm_softplus(false, X, X_mean, X_var, eps);
        Tensor tY3 = eg.batchNorm_gelu(false, X, X_mean, X_var, eps);
//        Tensor tY3 = eg.batchNorm_sigmoid(false, X, X_mean, X_var, eps);
//        Tensor tY3 = eg.batchNorm_tanh(false, X, X_mean, X_var, eps);
        
//        Tensor tdeltaX2 = eg.batchNorm_leakyRelu_deltaX_v2(true, deltaY, k, X, X_mean, X_var, eps);
//        Tensor tdeltaX2 = eg.batchNorm_elu_deltaX_v2(true, deltaY, alpha, k, X, X_mean, X_var, eps);
//        Tensor tdeltaX2 = eg.batchNorm_softplus_deltaX_v2(true, deltaY, X, X_mean, X_var, eps);
        Tensor tdeltaX2 = eg.batchNorm_gelu_deltaX_v2(false, deltaY, X, X_mean, X_var, eps);
//        Tensor tdeltaX2 = eg.batchNorm_sigmoid_deltaX_v2(true, deltaY, X, X_mean, X_var, eps);
//        Tensor tdeltaX2 = eg.batchNorm_tanh_deltaX_v2(false, deltaY, X, X_mean, X_var, eps);
        
        //compare---------------------------------------------------------------
        float[] Y1 = tY2.value(), Y2 = tY3.value();
        float[] dX1 = tdeltaX1.value(), dX2 = tdeltaX2.value();
        
        float sp0 = Vector.samePercent_relative(Y1, Y2);
        float sp1 = Vector.samePercent_relative(dX1, dX2);
        
        Vector.println("Y1 = ", Y1, 0, 10);
        Vector.println("Y2 = ", Y2, 0, 10);
        System.out.println("sp0 (Y) = " + sp0);
        
        Vector.println("dX1 = ", dX1, 0, 10);
        Vector.println("dX2 = ", dX2, 0, 10);
        System.out.println("sp1 (deltaX) = " + sp1);
 
        if(sp0 < 0.95f) throw new RuntimeException();
        if(sp1 < 0.95f) throw new RuntimeException();
        System.gc();
    }
    
    public static void main(String[] args) {
        Vector.PRINT_DIFFERENT = true;
        
        for(int h=2; h<=32; h++)
            for(int w=1; w<=256; w++) testCorrect(h, w);
            
        for(int h=100; h<=105; h++)
            for(int w= 128; w<=256; w++) testCorrect(h, w);
            
        for(int h=1000; h<=1028; h++)
            for(int w= 233; w<=256; w++) testCorrect(h, w);
    }
}
