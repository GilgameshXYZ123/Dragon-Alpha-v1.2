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
import z.dragon.nn.unit.dual.blas.BatchMatMulT2;
import z.util.math.ExRandom;
import z.util.math.vector.Matrix;
import z.util.math.vector.Vector;

/**
 *
 * @author Gilgamesh
 */
public class Cuda_function_test7 
{
    static final ExRandom exr = new ExRandom();
    static { alpha.home("C:\\Users\\Gilgamesh\\Desktop\\Dragon-alpha");}
    static Engine eg = alpha.engine.cuda_float32(0, alpha.engine.memp1());
    
    static float[][] mX1 = new float[][]{//7 * 3
        {-0.1f, 0.2f, -0.3f, 0.4f, -0.3f,  0.2f,  0.1f},
        { 0.5f, 0.5f,  0.4f, 0.5f,  0.2f,  0.9f,  0.2f},
        {-0.1f, 0.1f, -0.2f, 0.6f, -0.1f, -0.1f,  0.8f},
        { 0.1f, 0.2f,  0.3f, 0.4f,  0.3f, -0.2f, -0.1f},
        {-0.5f, 0.5f, -0.4f, 0.5f, -0.2f,  0.9f,  0.2f},
        { 0.1f, 0.1f,  0.2f, 0.6f,  0.1f,  0.1f, -0.8f},
        {-0.1f, 0.2f, -0.3f, 0.4f, -0.3f,  0.2f, -0.1f},
    };
    
    static float[][] mX2 = new float[][]{//7 * 3
        {-0.1f, 0.2f, -0.4f, 0.4f, -0.3f,  0.2f,  0.1f},
        { 0.5f, 0.5f,  0.4f, 0.5f,  0.2f,  0.9f,  0.2f},
        {-0.1f, 0.1f, -0.2f, 0.6f, -0.1f, -0.1f,  0.8f},
        { 0.1f, 0.2f,  0.7f, 0.4f,  0.3f, -0.2f, -0.1f},
        {-0.5f, 0.5f, -0.7f, 0.5f, -0.2f,  0.9f,  0.2f},
        { 0.1f, 0.7f,  0.2f, 0.6f,  0.1f,  0.1f, -0.1f},
        {-0.1f, 0.0f, -0.3f, 0.4f, -0.3f,  0.2f, -0.8f},
    };
    
    static int height = mX1.length;
    static int width = mX1[0].length;

    static Tensor tX1 = eg.tensor(mX1, 1, height, width);
    static Tensor tX2 = eg.tensor(mX2, 1, height, width);
        
    public static void test2()
    {
//        MatMul mt = nn.matMul().init(eg);
//        Tensor tY = mt.forward(tX1, tX2)[0];
//        Tensor[] tdeltaX = mt.backward(eg.ones_like(tY));
        
//        MatMulT1 mt1 = nn.matMulT1().init(eg);
//        Tensor tY = mt1.forward(tX1, tX2)[0];
//        Tensor[] tdeltaX = mt1.backward(eg.ones_like(tY));
        
//        MatMulT2 mt2 = nn.matMulT2().init(eg);
//        Tensor tY = mt2.forward(tX1, tX2)[0];
//        Tensor[] tdeltaX = mt2.backward(eg.ones_like(tY));
        
//        BatchMatMul bmt = nn.batchMatMul().init(eg);
//        Tensor tY = bmt.forward(tX1, tX2)[0];
//        Tensor[] tdeltaX = bmt.backward(eg.ones_like(tY));
        
//        BatchMatMulT1 bmt1 = nn.batchMatMulT1().init(eg);
//        Tensor tY = bmt1.forward(tX1, tX2)[0];
//        Tensor[] tdeltaX = bmt1.backward(eg.ones_like(tY));
        
        BatchMatMulT2 bmt2 = nn.batchMatMulT2().init(eg);
        Tensor tY = bmt2.forward(tX1, tX2)[0];
        Tensor[] tdeltaX = bmt2.backward(eg.ones_like(tY));
        
        //forward---------------------------------------------------------------
        float sum1 = eg.straight_sum(tY).get();
        System.out.println("sum = " + sum1);
        
        float[] Y = tY.value();
        float[][] mY = Vector.to2D(Y, height, width);
        Matrix.println(mY);
        
        //backward--------------------------------------------------------------
        Tensor tdeltaX1 = tdeltaX[0];
        Tensor tdeltaX2 = tdeltaX[1];
        
        float sum2 = eg.straight_sum(tdeltaX1).get();
        float sum3 = eg.straight_sum(tdeltaX2).get();
        
        float[] deltaX1 = tdeltaX1.value();
        float[] deltaX2 = tdeltaX2.value();
        float[][] mdeltaX1 = Vector.to2D(deltaX1, height, width);
        float[][] mdeltaX2 = Vector.to2D(deltaX2, height, width);
        
        System.out.println("sum2 = " + sum2);
        Matrix.println(mdeltaX1);
        
        System.out.println("sum3 = " + sum3);
        Matrix.println(mdeltaX2);
        
        if(Float.isNaN(sum1)) throw new NullPointerException();
        if(Float.isNaN(sum2)) throw new NullPointerException();
        if(Float.isNaN(sum3)) throw new NullPointerException();
    }
    
    public static void main(String[] args) 
    {
        test2();
    }
}
