/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package test.cuda;

import static z.dragon.alpha.Alpha.alpha;
import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.util.math.vector.Vector;

/**
 *
 * @author Gilgamesh
 */
public class Cuda_transpose4D_test
{
    static { alpha.home("C:\\Users\\Gilgamesh\\Desktop\\Dragon-alpha-v1.1");}
    static Engine eg = alpha.engine.cuda_float32(0, alpha.engine.memp1());
    
    public static void testCorrect(
            int dim0, int dim1, int dim2, int dim3,
            int dimIdx1, int dimIdx2) 
    {
        System.out.println("testCorrect: [dim0, dim1, dim2. dim3]");
        System.out.println(dim0 + ", " + dim1 + ", " + dim2 + ", " + dim3);
        System.out.println("dimIdx1 = " + dimIdx1);
        System.out.println("dimIdx2 = " + dimIdx2);
        
        float[] A = Vector.random_float_vector(dim0 * dim1 * dim2 * dim3, -1, 1);
        
        //CPU-------------------------------------------------------------------
        float[][][][] mA = Vector.to4D(A, dim0, dim1, dim2, dim3);
        float[][][][] mB = Vector.transpose4D(mA, dimIdx1, dimIdx2);
        float[] B1 = Vector.flatten(mB);
        
        System.out.print("CPU: "); Vector.println(B1, 0, 10);
        
        //GPU-------------------------------------------------------------------
        Tensor tA = eg.tensor(A, dim0, dim1, dim2, dim3).c();
        Tensor tB = eg.transpose(true, tA, dimIdx1, dimIdx2).c();
        
        float[] B2 = tB.value();
        System.out.print("GPU: "); Vector.println(B2, 0, 10);
        
        float zp1 = Vector.zeroPercent(B2);
        System.out.println("zp1 = " + zp1);
                
        //compare---------------------------------------------------------------
        float sp = Vector.samePercent_absolute(B1, B2);
        System.out.println("sp = " + sp);
        if(sp != 1) throw new RuntimeException();
        
        eg.delete(tA, tB);
    
    }
    public static void main(String[] args) 
    {
        for(int dim0 = 16; dim0 <= 32; dim0++)
            for(int dim1 = 4; dim1 <= 32; dim1++)
                for(int dim2 = 4; dim2 <= 32; dim2++)
                    for(int dim3 = 4; dim3 <= 32; dim3++)
                        testCorrect(dim0, dim1, dim2, dim3, 2, 3);
        
        //(0, 1): correct
//        testCorrect(16, 4, 13, 22, 1, 2);
                    //16, 4, 13, 19
                    //16, 4, 13, 18
                    //16, 4, 15, 20
                    //(1, 3): 16, 4, 12, 20
                    //(2, 3): 16, 4, 12, 32
        
//        int dim0 = 32, dim1 = 32, dim2 = 32, dim3 = 32;
//        testCorrect(dim0, dim1, dim2, dim3, 1, 2);
    }
}
