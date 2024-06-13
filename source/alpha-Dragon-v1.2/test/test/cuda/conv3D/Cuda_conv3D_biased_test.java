/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package test.cuda.conv3D;

import static z.dragon.alpha.Alpha.UnitFunctional.F;
import static z.dragon.alpha.Alpha.alpha;
import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.dragon.engine.cuda.impl.math.Cuda_conv3D;
import z.dragon.engine.memp.Mempool;


/**
 *
 * @author Gilgamesh
 */
public class Cuda_conv3D_biased_test 
{
    static { alpha.home("C:\\Users\\Gilgamesh\\Desktop\\Dragon-alpha-v1.2");}
    static Mempool memp = alpha.engine.memp1(alpha.MEM_1GB * 8);
    static Engine eg = alpha.engine.cuda_float32(0, memp, alpha.MEM_1MB * 2048);
    
    public static void testCorrect(
            int IH, int IW, 
            int FH, int FW, 
            int N, int IC, int OC,
            int sh, int sw, int ph, int pw)
    {     
        int[] ODim = Cuda_conv3D.output_feature_dim(IH, IW, FH, FW, N, OC, sh, sw, ph, pw);
        int OH = ODim[1], OW = ODim[2];
        
        System.out.println("Test correct:");
        System.out.format("\tXsize( N, IC, IH, IW) = (%d, %d, %d, %d)\n", N, IC, IH, IW);
        System.out.format("\tWsize(OC, IC, FH, FW) = (%d, %d, %d, %d)\n", OC, IC, FH, FW);
        System.out.format("\tYsize( N, OC, OH, OW) = (%d, %d, %d, %d)\n", N, OC, OH, OW);
        
        Tensor X = eg.Gaussian(N, IH, IW, IC);
        Tensor W = eg.Gaussian(OC, FH, FW, IC);
        Tensor B = eg.Gaussian(OC);
        
        Tensor tY1 = eg.conv3D(X, W, sh, sw, ph, pw);
        tY1 = eg.add_row(true, tY1, B);
        
        Tensor tY2 = eg.conv3D_biased(X, W, sh, sw, ph, pw, B);
        
        float sp = eg.straight_equal(tY1, tY2).get();
        
        System.out.println(sp);
        if(sp < 0.999f) throw new RuntimeException();
        
        float nzp1 = eg.nonzero_percent(tY1).get();
        float nzp2 = eg.nonzero_percent(tY2).get();
        
        System.out.println("nzp1 = " + nzp1);
        System.out.println("nzp2 = " + nzp2);
        
        System.gc();
    }
    
    public static void main(String[] args)
    {
//        testCorrect(128, 128, 3, 3, 256,  3,  64, 1, 1, 1, 1);//error(all 0s): [256, 128, 128,   4] -> [256, 128, 128,  64]
//        testCorrect(128, 128, 3, 3, 256,  64,  64, 1, 1, 1, 1);//error(all 0s): [256, 128, 128,  64] -> [256, 128, 128,  64]
        
        {
            int IH = 16, IW = 16;//(33 - 4 + 2)/1 + 1
            int FH = 3, FW = 3;
            int sh = 1, sw = 1, ph = 1, pw = 1;
            int N = 4;
            int IC = 16, OC = 128;//9*4=36 
            for(int oc = 1; oc <= 128; oc++) {
                testCorrect(IH, IW, FH, FW, N, IC, oc, sh, sw, ph, pw);
            }
        }
        
        {
            int IH = 16, IW = 16;//(33 - 4 + 2)/1 + 1
            int FH = 4, FW = 4;
            int sh = 2, sw = 2, ph = 1, pw = 1;
            int N = 4;
            int IC = 16, OC = 128;//9*4=36 
            for(int oc = 1; oc <= 128; oc++) {
                testCorrect(IH, IW, FH, FW, N, IC, oc, sh, sw, ph, pw);
            }
        }
        
        {
            int IH = 16, IW = 16;//(33 - 4 + 2)/1 + 1
            int FH = 5, FW = 5;
            int sh = 2, sw = 2, ph = 2, pw = 2;
            int N = 4;
            int IC = 16, OC = 128;//9*4=36 
            for(int oc = 1; oc <= 128; oc++) {
                testCorrect(IH, IW, FH, FW, N, IC, oc, sh, sw, ph, pw);
            }
        }
        
        {
            int IH = 8, IW = 8;//(33 - 4 + 2)/1 + 1
            int FH = 5, FW = 5;
            int sh = 2, sw = 2, ph = 2, pw = 2;
            int N = 64;
            int IC = 16, OC = 128;//9*4=36 
            for(int oc = 1; oc <= 128; oc++) {
                testCorrect(IH, IW, FH, FW, N, IC, oc, sh, sw, ph, pw);
            }
        }
    }

}
