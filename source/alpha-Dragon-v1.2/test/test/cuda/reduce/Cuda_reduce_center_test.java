/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package test.cuda.reduce;

import static z.dragon.alpha.Alpha.alpha;
import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.util.math.ExRandom;
import z.util.math.vector.Matrix;
import z.util.math.vector.Vector;

/**
 *
 * @author Gilgamesh
 */
public class Cuda_reduce_center_test 
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
        
        float[] X1 = Vector.random_float_vector(length, -1, 1);
        float[] X2 = Vector.random_float_vector(length, -1, 1);
        
        Tensor tX1 = eg.tensor(X1, dim0, dim1, dim2);
        Tensor tX2 = eg.tensor(X2, dim0, dim1, dim2);
        
        //CPU-------------------------------------------------------------------
        float[][][] cX1 = Vector.to3D(X1, dim0, dim1, dim2);
        float[][][] cX2 = Vector.to3D(X2, dim0, dim1, dim2);
        
        float[][] mY = Matrix.center_quadratic2(cX1, cX2, alpha, alpha2, beta, beta2, gamma, gamma2);
        float[] Y1 = Vector.flatten(mY);
        
        //GPU-------------------------------------------------------------------
        Tensor tY = eg.center_quadratic2(tX1, tX2, alpha, alpha2, beta, beta2, gamma, gamma2);
                
        //compare---------------------------------------------------------------
        float[] Y2 = tY.value();
        
        Vector.println("CPU1: ", Y1, 0, 10);
        Vector.println("GPU1:" , Y2, 0, 10);
        
        
        float sp = Vector.samePercent_relative(Y1, Y2);
        System.out.println("sp = " + sp);
        
        if(sp < 0.99f) throw new RuntimeException();
        System.gc();
    }
    
    
    public static void main(String[] args) {
        for(int d0=2; d0<=32; d0++)
        for(int d1=2; d1<=32; d1++)
        for(int d2=2; d2<=32; d2++)
            testCorrect(d0, d1, d2);
        
        for(int d0=41; d0<=52; d0++)
        for(int d1=41; d1<=52; d1++)
        for(int d2=41; d2<=52; d2++)
            testCorrect(d0, d1, d2);
        
        for(int d0=141; d0<=152; d0++)
        for(int d1=141; d1<=152; d1++)
        for(int d2=141; d2<=152; d2++)
            testCorrect(d0, d1, d2);
    }
} 
