/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package test.cuda.reduce;

import static z.dragon.alpha.Alpha.alpha;
import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.dragon.engine.cuda.impl.CudaException;
import z.util.math.ExRandom;
import z.util.math.vector.Vector;

/**
 *
 * @author Gilgamesh
 */
public class Cuda_reduce_field_max_test {
    static { alpha.home("C:\\Users\\Gilgamesh\\Desktop\\Dragon-alpha-v1.1");}
    static Engine eg = alpha.engine.cuda_float32(0, alpha.engine.memp1(alpha.MEM_1GB * 6));
    static final ExRandom exr = new ExRandom();

    public static void testCorrect(int height, int width) {
        Tensor X = eg.Gaussian(height, width);
        Tensor[] V  = eg.field_var_mean(false, X);
        System.out.println(V[0].sum());
        System.out.println(V[1].sum());
        System.out.println(V[0].zero_percent());
        System.out.println(V[1].zero_percent());
        System.out.println(X.zero_percent());
    }
    
    public static void main(String[] args)
    {
        try {
            testCorrect(256*128*128, 64);
        }
        catch(Exception e)
        {
            e.printStackTrace();
        }
        System.out.println(CudaException.lastException());
    }
}
