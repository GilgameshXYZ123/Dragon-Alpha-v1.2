/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package random;

import static z.dragon.alpha.Alpha.alpha;
import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.util.math.ExRandom;
import z.util.math.vector.Vector;

/**
 *
 * @author Gilgamesh
 */
public class r3 
{
    static { alpha.home("C:\\Users\\Gilgamesh\\Desktop\\Dragon-alpha-v1.1");}
    static Engine eg = alpha.engine.cuda_float32(0, alpha.engine.memp1());
    static final ExRandom exr = new ExRandom();
    
    public static void test1()
    {
        eg.engineCore().random().setSeed(12223);
        Tensor X = eg.Uniform(32);
         
        float[] v = X.value();
        Vector.println(v, 0, 10);
        
        float k = Vector.entropyE(v);
        System.out.println(k);
    }

    public static void main(String[] args)
    {
        test1();
    }
}
