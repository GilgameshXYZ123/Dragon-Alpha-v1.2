/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package test.summer;

import static z.dragon.alpha.Alpha.UnitBuilder.nn;
import static z.dragon.alpha.Alpha.UnitFunctional.F;
import static z.dragon.alpha.Alpha.alpha;
import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.dragon.nn.unit.complex.Module;
import z.dragon.nn.unit.simple.affine.SqBatchNorm;
import z.util.math.vector.Matrix;
import z.util.math.vector.Vector;

/**
 *
 * @author Gilgamesh
 */
public class BN_test2 
{
    static { alpha.home("C:\\Users\\Gilgamesh\\Desktop\\Dragon-alpha-v1.1");}
    static Engine eg = alpha.engine.cuda_float32(0, alpha.engine.memp1());
    
    public static float[] random1D(int length, float min, float max) 
    {
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
    
    public static float[][] random2D(int height, int width, float min, float max) 
    {
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
        
        System.out.println("X = " + tX.sum()); Matrix.println(X);
        System.out.print("A = " + tA.sum()); Vector.println(A);
        System.out.print("B = " + tB.sum()); Vector.println(B);
    
        //forward prop----------------------------------------------------------
        Module f = new Module() {
            SqBatchNorm bn = nn.sqBatchNorm(false, width); 
            
            @Override
            public void __init__(Engine eg) { bn.weight(tA).bias(tB); }
            
            @Override
            public Tensor[] __forward__(Tensor... X) {
                Tensor[] res = X;
                X = F.leakyRelu(false, X);
                X = bn.forward(X);
                X = F.leakyRelu(false, X);
                X = F.add(X[0], res[0]);
                return X;
            }
        }.init(eg);
        
        Tensor tY = f.forward(tX)[0];
                
        Tensor[] statsX = eg.field_var_mean(tX);
        Tensor Xvar = statsX[0];
        Tensor Xmean = statsX[1];
       
        System.out.println("Xmean = " + Xmean.sum()); Xmean.vprintln();
        System.out.println("Xvar = " + Xvar.sum()); Xvar.vprintln();
        
        System.out.println("Ymean = " + tY.mean()); 
        System.out.println("Yvar = " + tY.var());

        //backward prop----------------------------------------------------------
        Tensor grad = eg.ones_like(tY);
        
        Tensor tdeltaX = f.backward(grad)[0]; 
        
        SqBatchNorm bn = f.unit("bn");
        Tensor tdeltaA = bn.weight().grad();
        Tensor tdeltaB = bn.bias().grad();
        
        //compare---------------------------------------------------------------
        float[] Y = tY.value();
        float[][] mY = Vector.to2D(Y, height, width);
        System.out.println("sum_Y = " + eg.straight_sum(tY).get());
        Matrix.println(mY);
        System.out.println();
        
        System.out.println("mod_count = " + tX.mod_count());
        System.out.println("mod_count = " + tX.mod_count());
        System.out.println("Y.mean = " + tY.mean());
        System.out.println("Y.var = " + tY.var());
        System.out.println();
        
        
        float[] deltaX = tdeltaX.value();
        float[][] mdeltaX = Vector.to2D(deltaX, height, width);
        System.out.println("sum_deltaX = " + eg.straight_sum(tdeltaX).get());
        Matrix.println(mdeltaX);
        System.out.println();
        
        
        float[] deltaA = tdeltaA.value();
        float[] deltaB = tdeltaB.value();
        
        System.out.println("sum_deltaA = " + eg.straight_sum(tdeltaA).get());
        Vector.println(deltaA);
        System.out.println();
        
        System.out.println("sum_deltaB = " + eg.straight_sum(tdeltaB).get());
        Vector.println(deltaB);
        System.out.println();
    }
    
    public static void main(String[] args)
    {
        test1(10, 6);
    }
    
}
