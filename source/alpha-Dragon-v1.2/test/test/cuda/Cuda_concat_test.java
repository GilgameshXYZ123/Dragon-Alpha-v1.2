/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package test.cuda;

import static z.dragon.alpha.Alpha.alpha;
import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.dragon.engine.cuda.CudaFloat32EngineBase;
import z.util.lang.SimpleTimer;
import z.util.math.vector.Vector;

/**
 *
 * @author Gilgamesh
 */
public class Cuda_concat_test 
{
    static { alpha.home("C:\\Users\\Gilgamesh\\Desktop\\Dragon-alpha-v1.2");}
    static Engine eg = alpha.engine.cuda_float32(0, alpha.engine.memp1());
    static {
        CudaFloat32EngineBase cu32 = (CudaFloat32EngineBase) eg.engineBase();
//        cu32.dconv3D_deltaX_s1_useTexture(false);
        cu32.conv3D_deltaX_KernelSplit_useTexture(true);
    }
    
    public static void test1(int N, int IH, int IW)
    {
        Tensor[] X = new Tensor[N];
        for(int j=0; j<N; j++) X[j] = eg.Gaussian(IH, IW, 4);
        
        //dim0 = 660
        //dim1 = 682
        //dim2 = 669
        SimpleTimer timer = new SimpleTimer().record();
        int nIter = 1000;
        for(int i=0; i<nIter; i++)
        {
            Tensor Y = eg.concat(0, X).view(true, N, IH, IW, 4); 
            System.out.println(Y);
            eg.delete(Y);
        }
        long dif = timer.record().timeStamp_dif_millis();
        System.out.println(dif);
        
        eg.delete(X);
    }
    
    public static void testCorrect2(int dimIdx, int height, int width)
    {
        System.out.println("\n\nheight = " + height);
        System.out.println("width = " + width);
        
        int length = height * width;
        float[] X1 = Vector.random_float_vector(length);
        float[] X2 = Vector.random_float_vector(length);
        float[] X3 = Vector.random_float_vector(length);
        
        Tensor tX1 = eg.tensor(X1, height, width);
        Tensor tX2 = eg.tensor(X2, height, width);
        Tensor tX3 = eg.tensor(X3, height, width);
        
        Tensor tY = eg.concat(dimIdx, tX1, tX2, tX3);
        eg.delete(tX1, tX2, tX3);
        
        Tensor[] tXs = eg.chunk(tY, dimIdx, 3);
        tX1 = tXs[0];
        tX2 = tXs[1];
        tX3 = tXs[2];
        
        float[] Y1 = tX1.value();
        float[] Y2 = tX2.value();
        float[] Y3 = tX3.value();
        
        float sp1 = Vector.samePercent_absolute(X1, Y1);
        float sp2 = Vector.samePercent_absolute(X2, Y2);
        float sp3 = Vector.samePercent_absolute(X3, Y3);
        
        System.out.print("X1: "); Vector.println(X1, 0, 10);
        System.out.print("Y1: "); Vector.println(Y1, 0, 10);
        System.out.println("sp1 = " + sp1);
        
        System.out.print("X2: "); Vector.println(X2, 0, 10);
        System.out.print("Y2: "); Vector.println(Y2, 0, 10);
        System.out.println("sp2 = " + sp2);
        
        System.out.print("X3: "); Vector.println(X3, 0, 10);
        System.out.print("Y3: "); Vector.println(Y3, 0, 10);
        System.out.println("sp3 = " + sp3);
        
        eg.delete(tXs);
        if(sp1 != 1) throw new RuntimeException();
        if(sp2 != 1) throw new RuntimeException();
        if(sp3 != 1) throw new RuntimeException();
    }
    
    public static void main(String[] args)
    {
        int N = 128, IH = 128, IW = 128;
        test1(N, IH, IW);
        
        for(int h=1; h<=100; h++)
        for(int w=1; w<100; w++)
            testCorrect2(0, h, w);
    }
}
