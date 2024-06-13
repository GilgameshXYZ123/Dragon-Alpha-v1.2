/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package test.spring;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import static z.dragon.alpha.Alpha.UnitBuilder.nn;
import static z.dragon.alpha.Alpha.alpha;
import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.dragon.nn.optim.Optimizer;
import z.dragon.nn.unit.simple.blas.FullConnect;
import z.util.math.ExRandom;
import z.util.math.vector.Matrix;
import z.util.math.vector.Vector;

/**
 *
 * @author Gilgamesh
 */
public class Cuda_function_testB 
{
    static final ExRandom exr = new ExRandom();
    static { alpha.home("C:\\Users\\Gilgamesh\\Desktop\\Dragon-alpha-v1.1");}
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
    
    static float[][] mW = new float[][]{//7 * 3
        {-0.1f, 0.2f, -0.4f, 0.4f, -0.3f,  0.2f,  0.1f},
        { 0.5f, 0.5f,  0.4f, 0.5f,  0.2f,  0.9f,  0.2f},
        {-0.1f, 0.1f, -0.2f, 0.6f, -0.1f, -0.1f,  0.8f},
        { 0.1f, 0.2f,  0.7f, 0.4f,  0.3f, -0.2f, -0.1f},
        {-0.5f, 0.5f, -0.7f, 0.5f, -0.2f,  0.9f,  0.2f},
        { 0.1f, 0.7f,  0.2f, 0.6f,  0.1f,  0.1f, -0.1f},
        {-0.1f, 0.0f, -0.3f, 0.4f, -0.3f,  0.2f, -0.8f},
    };
    
    static float[] B = new float[] {
        -0.3f, 0.2f, -0.4f, 0.9f, -0.1f,  0.9f,  0.1f
    };
//    static { B = new float[7]; }
    
    static Tensor tX = eg.tensor(mX, 7, 7).need_grad(true);
    static Tensor tW = eg.tensor(mW, 7, 7).need_grad(true);
    static Tensor tB = eg.tensor(B, 7).need_grad(true);
        
    public static void test2()
    {
        List<Tensor> params = new ArrayList<>();
        params.add(tW); params.add(tB);
        
//        Optimizer opt = alpha.optim.Adam(params, 0.01f);
        
        FullConnect fc = nn.fullconnect(true, 7, 7).init(eg);
        fc.weight(tW); 
        fc.bias(tB);
        Tensor tY = fc.forward(tX)[0];
        
        Tensor tdeltaX = fc.backward(eg.ones_like(tY))[0];
        Tensor tdeltaW = fc.weight().grad();
        Tensor tdeltaB = fc.bias().grad();
        
        //compare---------------------------------------------------------------
        float sum1 = eg.straight_sum(tY).get();
        float sum2 = eg.straight_sum(tdeltaX).get();
        float sum3 = eg.straight_sum(tdeltaW).get();
        float sum4 = eg.straight_sum(tdeltaB).get();
        
        float[] Y = tY.value();
        float[] deltaX = tdeltaX.value();
        float[] deltaW = tdeltaW.value();
        float[] deltaB = tdeltaB.value();
        
        float[][] mY = Vector.to2D(Y, 7, 7);
        float[][] mdeltaX = Vector.to2D(deltaX, 7, 7);
        float[][] mdeltaW = Vector.to2D(deltaW, 7, 7);
        
//        System.out.println("sum_Y = " + sum1);
//        Matrix.println(mY);
//        
//        System.out.println("sum_deltaX = " + sum2);
//        Matrix.println(mdeltaX);
//        
//        System.out.println("sum_deltaW = " + sum3);
//        Matrix.println(mdeltaW);
//        
//        System.out.println("sum_deltaB = " + sum4);
//        Vector.println(deltaB);
        
        if(Float.isNaN(sum1)) throw new NullPointerException();
        if(Float.isNaN(sum2)) throw new NullPointerException();
        if(Float.isNaN(sum3)) throw new NullPointerException();
        if(Float.isNaN(sum4)) throw new NullPointerException();
        
//        opt.update().clear_grads();
        
        float[][] mW = Vector.to2D(tW.value(), 7, 7);
        Matrix.println(mW);
        System.out.println();
    }
    
    public static void main(String[] args) 
    {
        for(int i=0; i<10; i++)
        test2();
    }
}
