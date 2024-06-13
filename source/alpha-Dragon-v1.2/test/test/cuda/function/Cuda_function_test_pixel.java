/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package test.cuda.function;

import static z.dragon.alpha.Alpha.alpha;
import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.util.math.ExRandom;
import z.util.math.vector.Vector;

/**
 *
 * @author Gilgamesh
 */
public class Cuda_function_test_pixel 
{
    static final ExRandom exr = new ExRandom();
    static { alpha.home("C:\\Users\\Gilgamesh\\Desktop\\Dragon-alpha-v1.1");}
    static Engine eg = alpha.engine.cuda_float32(0, alpha.engine.memp1());
    
    public static void testCorrect1(int IH, int IW) {
        int length = IH*IW;
        byte[] X1 = Vector.random_byte_vector(length);
        
        Tensor tBX1 = eg.tensor_int8(X1, IH, IW);
        Tensor tX = eg.img.pixel_to_dtype(true, tBX1);
        Tensor tBX2 = eg.img.dtype_to_pixel(true, tX);
        
        byte[] X2 = tBX2.value_int8();
        
        System.out.println("X1: "); Vector.println(X1, 0, 10);
        System.out.println("X2: "); Vector.println(X2, 0, 10);
        float sp = Vector.samePercent_absolute(X1, X2);
        System.out.println("sp == " + sp);
        if(sp != 1) throw new RuntimeException();
    }
    
    public static void main(String[] args)
    {
        for(int ih=1; ih<=64; ih++)
        for(int iw=1; iw<=64; iw++)
            testCorrect1(ih, iw);
    }
}
