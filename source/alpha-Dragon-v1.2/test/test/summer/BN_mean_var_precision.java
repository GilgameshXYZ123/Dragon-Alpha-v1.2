/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package test.summer;

import static z.dragon.alpha.Alpha.alpha;
import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.util.math.vector.Vector;

/**
 *
 * @author Gilgamesh
 */
public class BN_mean_var_precision
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
    
    public static void test1(int height, int width) {
        float[][] X = random2D(height, width, 0, 1);
        Tensor tX = eg.tensor(X, width);
        
        System.out.println("X.mean = " + tX.mean());
        System.out.println("X.var = " + tX.var());
        
        alpha.print("X.mean[0]", eg.field_mean(tX).data());
        alpha.print("X.var[0]", eg.field_var(tX).data());
        
        Vector.println("X.mean[1]", eg.row_mean(tX).value(), 0, 10);
        Vector.println("X.var[1]", eg.row_var(tX).value(), 0, 10);
    }

    public static void main(String[] args)
    {
        test1(512*32*32, 128);
    }
}
