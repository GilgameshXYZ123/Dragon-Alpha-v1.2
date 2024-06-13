/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package X.dconv3DX.FFT;

import static z.dragon.alpha.Alpha.alpha;
import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.util.math.vector.Matrix;

/**
 *
 * @author Gilgamesh
 */
public class test 
{
    static { alpha.home("C:\\Users\\Gilgamesh\\Desktop\\Dragon-alpha");}
    static Engine eg = alpha.engine.cuda_float32(0, alpha.engine.memp1());
    
    public static void test1()
    {
        float[] X = {1, 2, 3, 5, 
                     4, 7, 9, 5, 
                     1, 4, 6, 7,
                     5, 4, 3, 7,
                     8, 7, 5, 1};
        float[] W = {1, 5, 4, 
                     3, 6, 8,
                     1, 5, 7};
        
        Tensor tX = eg.tensor(X, 1, 5, 4, 1);
        Tensor tW = eg.tensor(W, 1, 3, 3, 1);
        Tensor tY = eg.conv3D(tX, tW, 1, 1, 0, 0);
        System.out.println(tY);
        
        tY.toString();
        
        float[][] Y = tY.value2D(2);
        Matrix.println(Y);
    }
    
    public static void main(String[] args)
    {
        test1();
    }
}
