/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package test.spring;

import static test.spring.Cuda_function_testC.testCorrect;
import static z.dragon.alpha.Alpha.UnitBuilder.nn;
import static z.dragon.alpha.Alpha.alpha;
import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.dragon.nn.unit.simple.pool2d.AvgPool2D;
import z.dragon.nn.unit.simple.pool2d.MaxPool2D;
import z.util.math.vector.Matrix;
import z.util.math.vector.Vector;

/**
 *
 * @author Gilgamesh
 */
public class Cuda_function_testD 
{
    static { alpha.home("C:\\Users\\Gilgamesh\\Desktop\\Dragon-alpha");}
    static Engine eg = alpha.engine.cuda_float32(0, alpha.engine.memp1());
    
    //(2, 5, 5, 3) -> (2, 3, 3, 3)
    static float[][][][] X = new float[][][][] {
        {
            {{0.3f, -0.1f, 0.3f}, { 0.3f, -0.2f, 0.1f}, {0.4f, 0.2f, 0.1f}, {-0.4f, 0.2f, 0.9f}, {0.1f, 0.2f, 0.1f}},
            {{1.5f, -0.1f, 0.3f}, {-0.2f,  0.8f, 0.1f}, {0.1f, 0.2f, 0.1f}, { 0.2f, 0.2f, 0.9f}, {0.1f, 0.2f, 0.1f}},
            {{2.0f,  0.4f, 0.3f}, {-0.3f, -0.2f, 0.1f}, {0.2f, 0.3f, 0.2f}, {-0.4f, 0.5f, 0.9f}, {0.1f, 0.2f, 0.1f}},
            {{0.1f, -0.1f, 0.3f}, {-0.7f,  0.3f, 0.1f}, {0.3f, 0.7f, 0.1f}, { 0.1f, 0.4f, 0.9f}, {0.1f, 0.2f, 0.1f}},
            {{1.3f, -0.6f, 0.3f}, { 0.7f, -0.5f, 0.1f}, {0.4f, 0.2f, 0.9f}, {0.4f, 0.7f, 0.9f}, {0.1f, 0.2f, 0.1f}}
        },
        {
            {{1.0f, 0.1f, 0.3f}, {0.3f, 0.2f, 0.1f}, {0.4f, 0.2f, 0.1f}, {0.8f, 0.2f, 0.9f}, {0.1f, 0.2f, 0.1f}},
            {{1.0f, 0.1f, 0.3f}, {0.3f, 0.2f, 0.1f}, {0.4f, 0.1f, 0.1f}, {0.9f, 0.2f, 0.9f}, {0.1f, 0.2f, 0.1f}},
            {{1.0f, 0.5f, 0.3f}, {0.3f, 0.2f, 0.1f}, {0.4f, 0.9f, 0.1f}, {0.1f, 0.2f, 0.2f}, {0.1f, 0.2f, 0.1f}},
            {{1.0f, 0.2f, 0.3f}, {0.3f, 0.2f, 0.1f}, {0.1f, 0.2f, 0.1f}, {0.4f, 0.2f, 0.1f}, {0.1f, 0.2f, 0.1f}},
            {{1.0f, 0.1f, 0.3f}, {0.3f, 0.2f, 0.1f}, {0.4f, 0.2f, 0.1f}, {0.4f, 0.8f, 0.9f}, {0.1f, 0.2f, 0.1f}}
        }
    };
    
    static Tensor tX = eg.tensor(Vector.flatten(X), 2, 5, 5, 3);
    
    public static void testCorrect() 
    {
        //(5 - 3) / 2 + 1 = 2 / 2 + 1 = 2, ()
        //(5 - 3 + 2) / 2 + 1 = 4/2 + 1 = 2 + 1 = 3
        AvgPool2D p2d = nn.avgPool2D(3, 2, 1).init(eg);
//        MaxPool2D p2d = nn.maxPool2D(3, 2, 1).init(eg);
        Tensor tY = p2d.forward(tX)[0].c();
        Tensor tdeltaX = p2d.backward(eg.ones_like(tY).c())[0].c();
        
        float sum1 = eg.straight_sum(tY).get();
        float sum2 = eg.straight_sum(tdeltaX).get();
        
        float[][] mY = Vector.to2D(tY.value(), 18, 3);
        float[][] mdeltaX = Vector.to2D(tdeltaX.value(), 50, 3);
        
        System.out.println("sum(Y) = " + sum1);
        Matrix.println(mY);
        
        System.out.println("sum(deltaX) = " + sum2);
        Matrix.println(mdeltaX);
    }
    
    public static void main(String[] args)
    {
        testCorrect();
    }
}
