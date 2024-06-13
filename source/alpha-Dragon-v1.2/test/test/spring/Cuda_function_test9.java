/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package test.spring;

import static z.dragon.alpha.Alpha.UnitBuilder.nn;
import static z.dragon.alpha.Alpha.alpha;
import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.dragon.nn.unit.simple.affine.SqBatchNorm;
import z.util.math.ExRandom;
import z.util.math.vector.Matrix;
import z.util.math.vector.Vector;

/**
 *
 * @author Gilgamesh
 */
public class Cuda_function_test9 
{
    static final ExRandom exr = new ExRandom();
    static { alpha.home("C:\\Users\\Gilgamesh\\Desktop\\Dragon-alpha");}
    static Engine eg = alpha.engine.cuda_float32(0, alpha.engine.memp1());
    
    static float[][] mX = new float[][]{//7 * 3
        {-0.1f, 0.2f, -0.3f, 0.4f, -0.3f,  0.2f,  0.1f},
        { 0.5f, 0.5f,  0.4f, 0.5f,  0.2f,  0.9f,  0.2f},
        {-0.1f, 0.1f, -0.2f, 0.6f, -0.1f, -0.1f,  0.8f},
        { 0.1f, 0.2f,  0.3f, 0.4f,  0.3f, -0.2f, -0.1f},
        {-0.5f, 0.5f, -0.4f, 0.5f, -0.2f,  0.9f,  0.2f},
        { 0.1f, 0.1f,  0.2f, 0.6f,  0.1f,  0.1f, -0.8f},
        {-0.1f, 0.2f, -0.3f, 0.4f, -0.3f,  0.2f, -0.1f},
    };
    
    static float[] A = new float[] { -0.1f, 0.2f, -0.4f, 0.4f, -0.3f, 0.2f, 0.1f};
    static float[] B = new float[] { -0.3f, 0.2f, -0.4f, 0.9f, -0.1f, 0.9f, 0.1f };
    
    static { A = new float[]{1, 1, 1, 1, 1, 1, 1}; }
    static { B = new float[7]; }
    
    static Tensor tX = eg.tensor(mX, 7, 7).need_grad(true);
    static Tensor tA = eg.tensor(A, 7).need_grad(true);
    static Tensor tB = eg.tensor(B, 7).need_grad(true);
    
    public static void test2()
    {
        Tensor[] Ys = eg.field_var_mean_sqmean(tX);
        Vector.println("var = ", Ys[0].value(), 0, 10);
        Vector.println("mean = ", Ys[1].value(), 0, 10);
        Vector.println("squareMean = ", Ys[2].value(), 0, 10);
        
        SqBatchNorm bn = nn.sqBatchNorm(7).init(eg);
        bn.weight(tA).bias(tB);
        
        System.out.println("mod_count = " + tX.mod_count());
        Tensor tY = bn.forward(tX)[0].c();
        System.out.println("mod_count = " + tX.mod_count());
        
        System.out.println("Y.mean = " + tY.mean());
        System.out.println("Y.var = " + tY.var());
        
        Tensor tdeltaX = bn.backward(eg.ones_like(tY))[0].c();
        
        Tensor tdeltaA = bn.weight().grad();
        Tensor tdeltaB = bn.bias().grad();
        
        //compare---------------------------------------------------------------
        float[] Y = tY.value();
        float[] deltaX = tdeltaX.value();
        float[] deltaA = tdeltaA.value();
        float[] deltaB = tdeltaB.value();
        
        float[][] mY = Vector.to2D(Y, 7, 7);
        float[][] mdeltaX = Vector.to2D(deltaX, 7, 7);
       
        System.out.println("sum_Y = " + eg.straight_sum(tY).get());
        Matrix.println(mY);
        
        System.out.println("sum_deltaA = " + eg.straight_sum(tdeltaA).get());
        Vector.println(deltaA);
        
        System.out.println("sum_deltaB = " + eg.straight_sum(tdeltaB).get());
        Vector.println(deltaB);
        
        System.out.println("sum_deltaX = " + eg.straight_sum(tdeltaX).get());
        Matrix.println(mdeltaX);
    }
    
    public static void main(String[] args)
    {
        test2();
    }
}
