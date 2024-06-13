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
public class Cuda_fullconect_test 
{
    static { alpha.home("C:\\Users\\Gilgamesh\\Desktop\\Dragon-alpha-v1.1");}
    static Engine eg = alpha.engine.cuda_float32(0, alpha.engine.memp1());
    
    public static void testCorrect(int IH, int IW) {
        Tensor X = eg.Gaussian(IH, IW);
        Tensor W = eg.Gaussian(IW, IW);
        Tensor B = eg.Gaussian(IW);
        
        Tensor Y1 = eg.fullconnect(X, W);
        Y1 = eg.add_row(true, Y1, B);
        
        Tensor Y2 = eg.fullconnect(X, W, B);
        
        float[] v1 = Y1.value();
        float[] v2 = Y2.value();
        float sp = Vector.samePercent_absolute(v1, v2);
        
        Vector.println("v1 = ", v1, 0, 10);
        Vector.println("v2 = ", v2, 0, 10);
        System.out.println("sp = " + sp);
        
        if(sp != 1) throw new RuntimeException();
        
    }
    

    public static void main(String[] args)
    {
        for(int ih=1; ih<=128; ih++)
        for(int iw=1; iw<=128; iw++) 
            testCorrect(ih, iw);
    }
}
