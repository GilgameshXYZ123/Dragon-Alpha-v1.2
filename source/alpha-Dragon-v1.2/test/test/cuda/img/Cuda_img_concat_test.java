/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package test.cuda.img;

import static z.dragon.alpha.Alpha.alpha;
import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.util.math.vector.Vector;

/**
 *
 * @author Gilgamesh
 */
public class Cuda_img_concat_test 
{
    static { alpha.home("C:\\Users\\Gilgamesh\\Desktop\\Dragon-alpha-v1.1");}
    static Engine eg = alpha.engine.cuda_float32(0, alpha.engine.memp1());
    
    public static void testCorrect(int dimIdx, int height, int width)
    {
        System.out.println("\n\nheight = " + height);
        System.out.println("width = " + width);
        
        int length = height * width;
        byte[] X1 = Vector.random_byte_vector(length);
        byte[] X2 = Vector.random_byte_vector(length);
        byte[] X3 = Vector.random_byte_vector(length);
        
        Tensor tX1 = eg.tensor_int8(X1, height, width);
        Tensor tX2 = eg.tensor_int8(X2, height, width);
        Tensor tX3 = eg.tensor_int8(X3, height, width);
        
        Tensor tY = eg.img.concat(dimIdx, tX1, tX2, tX3);
        eg.delete(tX1, tX2, tX3);
        
        Tensor[] tXs = eg.img.split(tY, dimIdx, height, height, height);
        tX1 = tXs[0];
        tX2 = tXs[1];
        tX3 = tXs[2];
        
        byte[] Y1 = tX1.value_int8();
        byte[] Y2 = tX2.value_int8();
        byte[] Y3 = tX3.value_int8();
        
        float sp1 = Vector.samePercent_absolute(X1, Y1);
        float sp2 = Vector.samePercent_absolute(X2, Y2);
        float sp3 = Vector.samePercent_absolute(X3, Y3);
        
//        System.out.print("X1: "); Vector.println(X1, 0, 10);
//        System.out.print("Y1: "); Vector.println(Y1, 0, 10);
//        
//        System.out.print("X2: "); Vector.println(X2, 0, 10);
//        System.out.print("Y2: "); Vector.println(Y2, 0, 10);
//        
//        System.out.print("X3: "); Vector.println(X3, 0, 10);
//        System.out.print("Y3: "); Vector.println(Y3, 0, 10);
        
        System.out.println("sp1 = " + sp1);
        System.out.println("sp2 = " + sp2);
        System.out.println("sp3 = " + sp3);
        
        eg.delete(tXs);
        if(sp1 != 1) throw new RuntimeException();
        if(sp2 != 1) throw new RuntimeException();
        if(sp3 != 1) throw new RuntimeException();
    }
    
    public static void main(String[] args)
    {
//        testCorrect(0, 2, 65);
        
        try
        {
            for(int h=1; h<=200; h++)
            for(int w=1; w<200; w++)
                testCorrect(0, h, w);
        }
        catch(Exception e) {
            e.printStackTrace();
        }
       
    }
}
