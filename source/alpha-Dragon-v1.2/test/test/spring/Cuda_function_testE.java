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
import z.dragon.nn.unit.reducer.tensor.Concat;
import z.dragon.nn.unit.simple.math2.Sigmoid;
import z.util.math.ExRandom;
import z.util.math.vector.Matrix;

/**
 *
 * @author Gilgamesh
 */
public class Cuda_function_testE 
{
    static final ExRandom exr = new ExRandom();
    static { alpha.home("C:\\Users\\Gilgamesh\\Desktop\\Dragon-alpha");}
    static Engine eg = alpha.engine.cuda_float32(0, alpha.engine.memp1());
    
    static float[] A = new float[]{1, 2, 3, 1, 2, 3, 1, 2, 3, 0, 1, 2};//3 * 4
    static float[] B = new float[]{1, 2, 4, 1, 2, 4, 1, 2, 4, 9, 1, 1};//3 * 4
    static float[] C = new float[]{2, 3, 4, 2, 3, 4, 2, 3, 4, 5, 2, 7};//3 * 4
    static float[] D = new float[]{4, 2, 1, 4, 2, 1, 9, 9, 9, 1, 2, 8};//3 * 4
    
    static Tensor tA = eg.tensor(A, 1, 1, 4, 3);
    static Tensor tB = eg.tensor(B, 1, 1, 4, 3);
    static Tensor tC = eg.tensor(C, 1, 1, 4, 3);
    static Tensor tD = eg.tensor(D, 1, 1, 4, 3);
    
    public static void test1()
    {
        Concat ct = nn.concat(-1).init(eg);
        Sigmoid f = nn.sigmoid().init(eg);
        
        Tensor tX = ct.forward(tA, tB, tC, tD)[0];
        Tensor tY = f.forward(tX)[0];
        
        Tensor[] deltaX1 = f.backward(eg.ones_like(tX));
        Tensor[] deltaX2 = ct.backward(deltaX1);
        Tensor dA = deltaX2[0];
        Tensor dB = deltaX2[1];
        Tensor dC = deltaX2[2];
        Tensor dD = deltaX2[3];
        
        float[][] mA = dA.value2D();
        float[][] mB = dB.value2D();
        float[][] mC = dC.value2D();
        float[][] mD = dD.value2D();
        
        System.out.println("sumA = " + dA.sum());
        Matrix.println(mA);
        
        System.out.println("sumB = " + dB.sum());
        Matrix.println(mB);
        
        System.out.println("sumC = " + dC.sum());
        Matrix.println(mC);
        
        System.out.println("sumC = " + dD.sum());
        Matrix.println(mD);
    }
    
    public static void main(String[] args)
    {
        test1();
    }
}
