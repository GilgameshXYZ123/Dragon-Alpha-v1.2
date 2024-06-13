/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package test.spring;

import static z.dragon.alpha.Alpha.alpha;
import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.dragon.nn.loss.LossFunction;
import z.util.math.ExRandom;
import z.util.math.vector.Matrix;
import z.util.math.vector.Vector;


/**
 *
 * @author Gilgamesh
 */
public class Cuda_function_test4 
{
    static final ExRandom exr = new ExRandom();
    static { alpha.home("C:\\Users\\Gilgamesh\\Desktop\\Dragon-alpha");}
    static Engine eg = alpha.engine.cuda_float32(0, alpha.engine.memp1());
    
    static float[][] mX = new float[][]{//7 * 3
        {0.1f, 0.2f, 0.3f, 0.4f, 0.3f, 0.2f, 0.1f},
        {0.5f, 0.5f, 0.4f, 0.5f, 0.2f, 0.9f, 0.2f},
        {0.1f, 0.1f, 0.2f, 0.6f, 0.1f, 0.1f, 0.8f},
        {0.1f, 0.2f, 0.3f, 0.4f, 0.3f, 0.2f, 0.1f},
        {0.5f, 0.5f, 0.4f, 0.5f, 0.2f, 0.9f, 0.2f},
        {0.1f, 0.1f, 0.2f, 0.6f, 0.1f, 0.1f, 0.8f},
        {0.1f, 0.2f, 0.3f, 0.4f, 0.3f, 0.2f, 0.1f},
        {0.5f, 0.5f, 0.4f, 0.5f, 0.2f, 0.9f, 0.2f},
        {0.1f, 0.1f, 0.2f, 0.6f, 0.1f, 0.1f, 0.8f},
    };
    
    static float[][] mY = new float[][]{//7 * 3
        {0.3f, 0.1f, 0.3f, 0.6f, 0.3f, 0.2f, 0.1f},
        {0.5f, 0.1f, 0.4f, 0.6f, 0.8f, 0.9f, 0.2f},
        {0.1f, 0.1f, 0.2f, 0.6f, 0.1f, 0.1f, 0.3f}, 
        {0.3f, 0.1f, 0.3f, 0.6f, 0.3f, 0.2f, 0.1f},
        {0.5f, 0.1f, 0.4f, 0.6f, 0.8f, 0.9f, 0.2f},
        {0.1f, 0.1f, 0.2f, 0.6f, 0.1f, 0.1f, 0.3f}, 
        {0.3f, 0.1f, 0.3f, 0.6f, 0.3f, 0.2f, 0.1f},
        {0.5f, 0.1f, 0.4f, 0.6f, 0.8f, 0.9f, 0.2f},
        {0.1f, 0.1f, 0.2f, 0.6f, 0.1f, 0.1f, 0.3f}
    };
    
    static float[][] mW = new float[][]{
        {0.1f, 0.2f, 0.3f},
        {0.6f, 0.9f, 0.2f},
        {0.5f, 0.2f, 0.4f}
    };
            
    static int height = mX.length;
    static int width = mX[0].length;

    static Tensor tX = eg.tensor(mX, height, width);
    static Tensor tY = eg.tensor(mY, height, width);
    static Tensor tW = eg.tensor(mW, 3, 3);
    
            
    public static void test2()
    {
        //L1: length
        //L2: length
        //SmoothL1: length
        //crossEntropy: firstDim = batch, 
//        LossFunction loss_func = alpha.loss.L1();
//        LossFunction loss_func = alpha.loss.L2();
//        LossFunction loss_func = alpha.loss.SmoothL1();
//        LossFunction loss_func = alpha.loss.binaryCrossEntropy();
        LossFunction loss_func = alpha.loss.softmax_crossEntropy(7);
        
        float ls = loss_func.loss(tX, tY).get();
        Tensor tdeltaX = loss_func.gradient(tX, tY).c();
        
        float[] deltaX = tdeltaX.value();
        float[][] mdeltaX = Vector.to2D(deltaX, height, width);
       
        System.out.println("ls = " + ls);
        System.out.println("sum(deltaX) = " + eg.straight_sum(tdeltaX).get());
        Matrix.println(mdeltaX);
    }
    
    public static void main(String[] args)
    {
        test2();
    }
}
