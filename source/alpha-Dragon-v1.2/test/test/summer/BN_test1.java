/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package test.summer;

import static z.dragon.alpha.Alpha.UnitBuilder.nn;
import static z.dragon.alpha.Alpha.alpha;
import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.dragon.nn.unit.simple.affine.SqBatchNorm;
import z.dragon.nn.unit.simple.math2.Sigmoid;
import z.util.math.vector.Matrix;
import z.util.math.vector.Vector;

/**
 *
 * @author Gilgamesh
 */
public class BN_test1 
{
    static { alpha.home("C:\\Users\\Gilgamesh\\Desktop\\Dragon-alpha-v1.1");}
    static Engine eg = alpha.engine.cuda_float32(0, alpha.engine.memp1());
    
    public static float[] random1D(int length, float min, float max) {
        if(min > max) { float t = min; min = max; max = t;}
        float div = max - min;
        
        int seed = 123;
        float[] X = new float[length];
        for(int i=0; i<length; i++) {
            seed = ((632229*seed) + 2100473) & 4194303;
            float v = min + div * (seed / 4194304.0f);
            X[i] = v;
        }
        return X;
    }
    
    public static float[][] random2D(int height, int width, float min, float max) {
        if(min > max) { float t = min; min = max; max = t;}
        float div = max - min;
        
        int seed = 123;
        float[][] X = new float[height][width];
        for(int i=0; i<height; i++)
            for(int j=0; j< width; j++) {
                seed = ((632229*seed) + 2100473) & 4194303;
                float v = min + div * (seed / 4194304.0f);
                X[i][j] = v;
            }
        
        return X;
    }
    
    public static void test1(int height, int width)
    {
        //prepare Area----------------------------------------------------------
        float[][] X = random2D(height, width, 0, 1);
        float[] A = random1D(width, 0, 1);
        float[] B = random1D(width, 0, 1);
        
        Tensor tX = eg.tensor(X, width).need_grad(true);
        Tensor tA = eg.tensor(A, width).need_grad(true);
        Tensor tB = eg.tensor(B, width).need_grad(true);
        
        System.out.println("X = "); Matrix.println(X);
        System.out.print("A = "); Vector.println(A);
        System.out.print("B = "); Vector.println(B);
        
        //forward prop----------------------------------------------------------
        SqBatchNorm bn = nn.sqBatchNorm(false, width).init(eg); bn.weight(tA).bias(tB);
        Sigmoid sg = nn.sigmoid(false).init(eg);
        
        Tensor tY = bn.forward(tX)[0];
        Tensor tZ = sg.forward(tY)[0];
                
        Tensor[] statX = eg.field_var_mean(tX);
        Tensor[] statY = eg.field_var_mean(tY);

        //backward prop----------------------------------------------------------
//        Tensor grad = eg.smul(false, tY, 10);//tY * tY * 10
        Tensor grad = eg.ones_like(tY);
        
        Tensor tdeltaY = sg.backward(grad)[0];
        Tensor tdeltaX = bn.backward(tdeltaY)[0]; 
        
        Tensor tdeltaA = bn.weight().grad();
        Tensor tdeltaB = bn.bias().grad();
        
        //compare---------------------------------------------------------------
        float[] Y = tY.value();
        float[] deltaX = tdeltaX.value();
        float[] deltaA = tdeltaA.value();
        float[] deltaB = tdeltaB.value();
        
        float[][] mY = Vector.to2D(Y, height, width);
        float[][] mdeltaX = Vector.to2D(deltaX, height, width);
        
        System.out.println("mod_count = " + tX.mod_count());
        System.out.println("mod_count = " + tX.mod_count());
        System.out.println("Y.mean = " + tY.mean());
        System.out.println("Y.var = " + tY.var());
        System.out.println();
       
        System.out.println("sum_Y = " + eg.straight_sum(tY).get());
        Matrix.println(mY);
        System.out.println();
        
        System.out.println("sum_deltaA = " + eg.straight_sum(tdeltaA).get());
        Vector.println(deltaA);
        System.out.println();
        
        System.out.println("sum_deltaB = " + eg.straight_sum(tdeltaB).get());
        Vector.println(deltaB);
        System.out.println();
        
        System.out.println("sum_deltaX = " + eg.straight_sum(tdeltaX).get());
        Matrix.println(mdeltaX);
        System.out.println();
    }
    
    public static void main(String[] args)
    {
        test1(8, 4);
    }
    
}
