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
public class Cuda_function_center_test 
{
    static { alpha.home("C:\\Users\\Gilgamesh\\Desktop\\Dragon-alpha-v1.1");}
    static Engine eg = alpha.engine.cuda_float32(0, alpha.engine.memp1());
    static final ExRandom exr = new ExRandom();
    
    public static void testCorrect(int dim0, int dim1, int dim2) {
        int length = dim0 * dim1 * dim2;
        int width = dim0 * dim2;
        
        System.out.format("\ntest_correct: (%d, %d, %d)\n", dim0, dim1, dim2);
        System.out.format("length = (%d, %d)\n", length, width);
        
        float alpha = exr.nextFloat(), alpha2 = exr.nextFloat();
        float beta = exr.nextFloat(), beta2 = exr.nextFloat();
        float gamma = exr.nextFloat(), gamma2 = exr.nextFloat();
        
//        System.out.println("alpha = " + alpha);
//        System.out.println("beta = " + beta);
//        System.out.println("gamma = " + gamma);
        
        float[] X1 = Vector.random_float_vector(length, -1, 1);
        float[] X2 = Vector.random_float_vector(width, -1, 1);
        
        System.out.print("X1: "); Vector.println(X1, 0, 10);
        System.out.print("X2: "); Vector.println(X2, 0, 10);
        
        Tensor dX1 = eg.tensor(X1, dim0, dim1, dim2);
        Tensor dX2 = eg.tensor(X2, dim0, dim2);
        
        //CPU-------------------------------------------------------------------
        float[][][] cX1 = Vector.to3D(X1, dim0, dim1, dim2);
        float[][] cX2 = Vector.to2D(X2, dim0, dim2);
        float[][][] cY = Vector.linear_center(cX1, cX2, alpha, beta, gamma);

//        float[][][] cY = Vector.quadratic2_center(cX1, cX2, alpha, alpha2, beta, beta2, gamma, gamma2);
        
        float[] Y1 = Vector.flatten(cY);
        System.out.print("CPU1: "); Vector.println(Y1, 0, 10);
        
        //GPU-------------------------------------------------------------------
        Tensor tY1 = eg.linear2_center(false, dX1, dX2, alpha, beta, gamma).c();
        Tensor tY2 = eg.linear2_center(true,  dX1, dX2, alpha, beta, gamma).c();

//        Tensor tY1 = eg.quadratic2_center(false, dX1, dX2, alpha, alpha2, beta, beta2, gamma, gamma2);
//        Tensor tY2 = eg.quadratic2_center(true, dX1, dX2, alpha, alpha2, beta, beta2, gamma, gamma2);
        
        //compare---------------------------------------------------------------
        float[] Y2 = tY1.value();
        System.out.print("GPU1: "); Vector.println(Y2, 0, 10);
        float sp1 = Vector.samePercent_relative(Y1, Y2); System.out.println("sp1:" + sp1);
        
        float[] Y3 = tY2.value();
        System.out.print("GPU2: "); Vector.println(Y3, 0, 10);
        float sp2 = Vector.samePercent_relative(Y1, Y3); System.out.println("sp2:" + sp2);
        
        if(sp1 < 0.99) throw new RuntimeException();
        if(sp2 < 0.99) throw new RuntimeException();
        System.gc();
    }
    
    public static void main(String[] args) {
        Vector.PRINT_DIFFERENT = true;
        
//        for(int d0=1; d0<=32; d0++)
//        for(int d1=1; d1<=32; d1++)
//        for(int d2=1; d2<=32; d2++)
//            testCorrect(d0, d1, d2);
//        
//        for(int d0=41; d0<=52; d0++)
//        for(int d1=41; d1<=52; d1++)
//        for(int d2=41; d2<=52; d2++)
//            testCorrect(d0, d1, d2);
        
        for(int d0=141; d0<=152; d0++)
        for(int d1=141; d1<=152; d1++)
        for(int d2=141; d2<=152; d2++)
            testCorrect(d0, d1, d2);
    }
}
