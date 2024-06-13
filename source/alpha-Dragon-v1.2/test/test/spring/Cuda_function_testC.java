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
import z.dragon.nn.unit.simple.blas.Conv3D;
import z.util.math.vector.Matrix;
import z.util.math.vector.Vector;

/**
 *
 * @author Gilgamesh
 */
public class Cuda_function_testC 
{
    static { alpha.home("C:\\Users\\Gilgamesh\\Desktop\\Dragon-alpha");}
    static Engine eg = alpha.engine.cuda_float32(0, alpha.engine.memp1());
    
    //(2, 5, 5, 3) -> (2, 3, 3, 3)
    static float[][][][] X = new float[][][][] {
        {
            {{1.0f, 0.1f, 0.3f}, {0.3f, 0.2f, 0.1f}, {0.4f, 0.2f, 0.1f}, {0.4f, 0.2f, 0.9f}, {0.1f, 0.2f, 0.1f}},
            {{1.0f, 0.1f, 0.3f}, {0.3f, 0.2f, 0.1f}, {0.4f, 0.2f, 0.1f}, {0.2f, 0.2f, 0.9f}, {0.1f, 0.2f, 0.1f}},
            {{1.0f, 0.4f, 0.3f}, {0.3f, 0.2f, 0.1f}, {0.4f, 0.2f, 0.1f}, {0.4f, 0.5f, 0.9f}, {0.1f, 0.2f, 0.1f}},
            {{1.0f, 0.1f, 0.3f}, {0.3f, 0.2f, 0.1f}, {0.4f, 0.2f, 0.1f}, {0.1f, 0.4f, 0.9f}, {0.1f, 0.2f, 0.1f}},
            {{1.0f, 0.6f, 0.3f}, {0.3f, 0.2f, 0.1f}, {0.4f, 0.2f, 0.1f}, {0.4f, 0.7f, 0.9f}, {0.1f, 0.2f, 0.1f}}
        },
        {
            {{1.0f, 0.1f, 0.3f}, {0.3f, 0.2f, 0.1f}, {0.4f, 0.2f, 0.1f}, {0.8f, 0.2f, 0.9f}, {0.1f, 0.2f, 0.1f}},
            {{1.0f, 0.1f, 0.3f}, {0.3f, 0.2f, 0.1f}, {0.4f, 0.1f, 0.1f}, {0.9f, 0.2f, 0.9f}, {0.1f, 0.2f, 0.1f}},
            {{1.0f, 0.5f, 0.3f}, {0.3f, 0.2f, 0.1f}, {0.4f, 0.9f, 0.1f}, {0.1f, 0.2f, 0.2f}, {0.1f, 0.2f, 0.1f}},
            {{1.0f, 0.2f, 0.3f}, {0.3f, 0.2f, 0.1f}, {0.1f, 0.2f, 0.1f}, {0.4f, 0.2f, 0.1f}, {0.1f, 0.2f, 0.1f}},
            {{1.0f, 0.1f, 0.3f}, {0.3f, 0.2f, 0.1f}, {0.4f, 0.2f, 0.1f}, {0.4f, 0.8f, 0.9f}, {0.1f, 0.2f, 0.1f}}
        }
    };
    
    //(3, 3, 3, 3)
    //(5 - 3 + 2*1) / 2 +  1 = 2 + 1 = 3
    static float[][][][] W = new float[][][][] {
        {
            {{1.0f, 0.1f, 0.3f}, {0.3f, 0.2f, 0.1f}, {0.4f, 0.2f, 0.1f}},
            {{1.0f, 0.1f, 0.3f}, {0.3f, 0.2f, 0.1f}, {0.4f, 0.2f, 0.1f}},
            {{1.0f, 0.1f, 0.3f}, {0.3f, 0.2f, 0.1f}, {0.4f, 0.2f, 0.1f}},
        },
        {
            {{1.0f, 0.1f, 0.3f}, {0.3f, 0.2f, 0.1f}, {0.4f, 0.2f, 0.1f}},
            {{1.0f, 0.1f, 0.3f}, {0.3f, 0.2f, 0.1f}, {0.4f, 0.2f, 0.1f}},
            {{1.0f, 0.1f, 0.3f}, {0.3f, 0.2f, 0.1f}, {0.4f, 0.2f, 0.1f}},
        },
        {
            {{1.0f, 0.1f, 0.3f}, {0.3f, 0.2f, 0.1f}, {0.4f, 0.2f, 0.1f}},
            {{1.0f, 0.1f, 0.3f}, {0.3f, 0.2f, 0.1f}, {0.4f, 0.2f, 0.1f}},
            {{1.0f, 0.1f, 0.3f}, {0.3f, 0.2f, 0.1f}, {0.4f, 0.2f, 0.1f}},
        }
    };
    
    static float[] B = new float[] {0.2f, -0.1f, 0.3f };
    
    static Tensor tX = eg.tensor(Vector.flatten(X), 2, 5, 5, 3);
    static Tensor tW = eg.tensor(Vector.flatten(W), 3, 3, 3, 3).need_grad(true);
    static Tensor tB = eg.tensor(B, 3).need_grad(true);

    public static void testCorrect() 
    {
        Conv3D conv3d = nn.conv3D(true, 3, 3, 3, 2, 1).init(eg);
        conv3d.bias(tB).weight(tW);
        
        Tensor tY = conv3d.forward(tX)[0];
        Tensor tdeltaX = conv3d.backward(eg.ones_like(tY).c())[0];
        
        Tensor tdeltaW = conv3d.weight().grad();
        Tensor tdeltaB = conv3d.bias().grad();
        
        float sum1 = eg.straight_sum(tY).get();
        float sum2 = eg.straight_sum(tdeltaX).get();
        float sum3 = eg.straight_sum(tdeltaW).get();
        float sum4 = eg.straight_sum(tdeltaB).get();
       
        float[][] mY = Vector.to2D(tY.value(), 18, 3);
        float[][] mdeltaX = Vector.to2D(tdeltaX.value(), 50, 3);
        float[][] mdeltaW = Vector.to2D(tdeltaW.value(), 9, 3);
        float[] deltaB = tdeltaB.value();
        
        System.out.println("sum(Y) = " + sum1);
        Matrix.println(mY);
        
        System.out.println("sum(deltaX) = " + sum2);
        Matrix.println(mdeltaX);
        
        System.out.println("sum(deltaW) = " + sum3);
        Matrix.println(mdeltaW);
        
        System.out.println("sum(deltaB) = " + sum4);
        Vector.println(deltaB);
    }
    
    public static void main(String[] args)
    {
        testCorrect();
    }
}
