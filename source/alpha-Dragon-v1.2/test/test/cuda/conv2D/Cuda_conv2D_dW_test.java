package test.cuda.conv2D;


import static z.dragon.alpha.Alpha.alpha;
import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.dragon.engine.cuda.CudaFloat32EngineBase;
import z.dragon.engine.cuda.impl.Cuda;
import z.util.lang.SimpleTimer;
import z.util.math.vector.Tense;
import z.util.math.vector.Vector;

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/**
 *
 * @author Gilgamesh
 */
public class Cuda_conv2D_dW_test 
{
    static { alpha.home("C:\\Users\\Gilgamesh\\Desktop\\Dragon-alpha-v1.2");}
    static Engine eg = alpha.engine.cuda_float32(0, alpha.engine.memp1());
    
    public static void testCorrect(
            int IW, int OW, int FW,
            int N, int IC, int OC,
            int sw, int pw)
    {
        if(IW == -1) IW = (OW - 1)*sw + FW - 2*pw;
        int OWp = OW + (OW-1) * (sw-1);
        int opw = pw;
        
        System.out.println("\nTest correct:");
        System.out.format("\tYsize( N, OC, OW) = (%d, %d, %d)\n", N, OC, OW);
        System.out.format("\tXsize( N, IC, IW) = (%d, %d, %d)\n", N, IC, IW);
        System.out.format("\tWsize(OC, IC, FW) = (%d, %d, %d)\n", OC, IC, FW);
        System.out.format("\t(sw, pw) = (%d, %d)\n", sw, pw);
        System.out.format("\t(OW_p) = (%d)\n", OWp);
        System.out.format("\t(oph, opw) = (%d)\n", opw);
        
        int GN = OC;
        int GM = IC * FW;
        int GK0 = N * OWp;
        System.out.format("(GN, GM, GK0) = (%d, %d, %d)\n", GN, GM, GK0);
        
        int sizeX = N * IW * IC;
        int sizeW = OC * FW * IC;
        int sizeY = N * OW * OC;
        
        float[] deltaY = Vector.random_float_vector(sizeY);
        float[] X = Vector.random_float_vector(sizeX);
        
        //CPU-------------------------------------------------------------------
        float[][][] cdeltaY = Vector.to3D(deltaY, N, OW, OC);
        float[][][] cX = Vector.to3D(X, N, IW, IC);
        float[][][] cdeltaW = new float[OC][FW][IC];
        
        Tense.deconv2D_deltaW_img2col2(
                cX,      IW, 
                cdeltaY, OW,
                cdeltaW, FW,
                N, IC, OC,
                sw, pw);
                
        float[] deltaW1 = Vector.flatten(cdeltaW);
        System.out.print("CPU1: ");Vector.println(deltaW1, 0, 10);
        float zp0 = Vector.zeroPercent(deltaW1); 
        
        //GPU-------------------------------------------------------------------
        Tensor tX = eg.tensor(X, N, IW, IC);
        Tensor tdeltaY  = eg.tensor(deltaY, N, OW, OC);
        Tensor tdeltaW1 = eg.conv2D_deltaW(tX, tdeltaY, FW, sw, pw).c();
        Tensor tdeltaW2 = eg.conv2D_deltaW(eg.empty(OC, FW, IC).c(), tX, tdeltaY, sw).c();
        
        float[] deltaW2 = eg.valueOf(tdeltaW1); System.out.print("GPU1: "); Vector.println(deltaW2, 0, 10);
        float[] deltaW3 = eg.valueOf(tdeltaW2); System.out.print("GPU2: "); Vector.println(deltaW3, 0, 10);
        float zp3 = Vector.zeroPercent(deltaW2);
        
        //compare---------------------------------------------------------------
        float sp1 = Vector.samePercent_relative(deltaW1, deltaW2, 1e-3f); System.out.println("sp1: "+sp1);
        float sp2 = Vector.samePercent_relative(deltaW1, deltaW3, 1e-3f); System.out.println("sp2: "+sp2);
        System.out.println("zp0: " + zp0);
        System.out.println("zp3: " + zp3);
        
        if(sp1 != 1) {throw new RuntimeException();}
        if(sp2 != 1) {throw new RuntimeException();}
        
        eg.delete(tX, tdeltaY, tdeltaW1, tdeltaW2);
    }
    
    public static void main(String[] args) {
        //test W11--------------------------------------------------------------
        {
            int OW = 15;
            int FW = 1;
            int N = 4, IC = 128;
            int sw = 1, pw = 0;
            for (int oc = 1; oc <= 128; oc++) 
                testCorrect(-1, OW, FW, N, IC, oc, sw, pw);
        }

        
        //test GemmSK W11-------------------------------------------------------
        {
            int OW = 15;
            int FW = 1;
            int N = 32, IC = 128;
            int sw = 1, pw = 0;
            for (int oc = 1; oc <= 128; oc++) 
                testCorrect(-1, OW, FW, N, IC, oc, sw, pw);

        }

        //test gemm-------------------------------------------------------------
        {
            int OW = 15;
            int FW = 4;
            int N = 4, IC = 32;
            int sw = 2, pw = 1;
            for(int oc = 1; oc <= 128; oc++)
                testCorrect(-1, OW, FW, N, IC, oc, sw, pw);
        }
        
        //test GemmSK-----------------------------------------------------------
        {
            int OW = 16;
            int FW = 4;
            int sw = 2, pw = 1;
            int N = 32, IC = 32;
            for(int oc = 1; oc <=128; oc++) 
                testCorrect(-1, OW, FW, N, IC, oc, sw, pw);
        }
    }
}
