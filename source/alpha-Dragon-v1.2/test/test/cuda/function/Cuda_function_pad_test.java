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
public class Cuda_function_pad_test 
{
    static final ExRandom exr = new ExRandom();
    static { alpha.home("C:\\Users\\Gilgamesh\\Desktop\\Dragon-alpha-v1.1");}
    static Engine eg = alpha.engine.cuda_float32(0, alpha.engine.memp1());
    
    public static void testCorrect4D(int N, int IH, int IW, int IC, int[] p0, int[] p1)
    {
        Tensor tX1 = eg.Gaussian(N, IH, IW, IC); System.out.println(tX1);
        
//        Tensor tY = eg.pad(false, tX1, p0, p1); System.out.println(tY);
//        Tensor tX2 = eg.trim(false, tY, p0, p1); System.out.println(tX2);

        int ON =  N + p0[0] + p1[0];
        int OH = IH + p0[1] + p1[1];
        int OW = IW + p0[2] + p1[2];
        int OC = IC + p0[3] + p1[3];
        Tensor tY = eg.expand(false, tX1, p0, new int[]{ ON, OH, OW, OC }); System.out.println(tY);
        Tensor tX2 = eg.crop(false, tY, p0, tX1.dim()); System.out.println(tX2);
        
        float[] X1 = tX1.value();
        float[] X2 = tX2.value();
        
        System.out.print("X1: "); Vector.println(X1, 0, 10);
        System.out.print("X2: "); Vector.println(X2, 0, 10);
         
        float sp = Vector.samePercent_absolute(X1, X2);
        System.out.println("sp = " + sp);
        if(sp != 1) throw new RuntimeException();
        
        tX1.delete();
        tY.delete();
        tX2.delete();
    }
    
    public static void testCorrect3D(int N,int IW, int IC, int[] p0, int[] p1)
    {
        Tensor tX1 = eg.Gaussian(N, IW, IC); System.out.println(tX1);
        Tensor tY = eg.pad(false, tX1, p0, p1); System.out.println(tY);
        Tensor tX2 = eg.trim(false, tY, p0, p1); System.out.println(tX2);
        
        float[] X1 = tX1.value();
        float[] X2 = tX2.value();
        
        System.out.print("X1: "); Vector.println(X1, 0, 10);
        System.out.print("X2: "); Vector.println(X2, 0, 10);
         
        float sp = Vector.samePercent_absolute(X1, X2);
        System.out.println("sp = " + sp);
        if(sp != 1) System.out.println("error");
        
        tX1.delete();
        tY.delete();
        tX2.delete();
    }
    
    public static void testCorrect2D(int N, int IC, int[] p0, int[] p1)
    {
        Tensor tX1 = eg.Gaussian(N, IC); System.out.println(tX1);
        Tensor tY = eg.pad(false, tX1, p0, p1); System.out.println(tY);
        Tensor tX2 = eg.trim(false, tY, p0, p1); System.out.println(tX2);
        
        float[] X1 = tX1.value();
        float[] X2 = tX2.value();
        
        System.out.print("X1: "); Vector.println(X1, 0, 10);
        System.out.print("X2: "); Vector.println(X2, 0, 10);
         
        float sp = Vector.samePercent_absolute(X1, X2);
        System.out.println("sp = " + sp);
        if(sp != 1) throw new RuntimeException();
        
        tX1.delete();
        tY.delete();
        tX2.delete();
    }
    
    public static void main(String[] args)
    {
        for(int n=1; n<=16; n++)
        for(int ih=1; ih<16; ih++)
        for(int iw=1; iw<16; iw++)
        for(int ic=1; ic<16; ic++)
            testCorrect4D(n, ih, iw, ic,
                    new int[] {0, 1, 2, 1},
                    new int[] {0, 1, 2, 1});
        
        
//        testCorrect3D(6, 11, 15,
//                    new int[] {1, 1, 1},
//                    new int[] {1, 1, 1}) ;
        
//        for(int n=1; n<=24; n++)
//        for(int iw=1; iw<24; iw++)
//        for(int ic=1; ic<24; ic++)
//            testCorrect3D(n, iw, ic,
//                    new int[] {1, 2, 1},
//                    new int[] {1, 2, 1});
        
//        for(int n=1; n<=64; n++)
//        for(int ic=1; ic<64; ic++)
//            testCorrect2D(n, ic,
//                    new int[] {1, 0},
//                    new int[] {1, 0});
        
    }
}
