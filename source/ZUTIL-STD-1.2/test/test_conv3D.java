/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package X.dconv3DX.FFT;

import z.dragon.engine.cuda.impl.math.Cuda_conv3D;
import z.util.math.vector.Tense;
import z.util.math.vector.Vector;

/**
 *
 * @author Gilgamesh
 */
public class test_conv3D 
{
    public static void testCorrect(
            int IH, int IW, 
            int FH, int FW, 
            int N, int IC, int OC,
            int sh, int sw, int ph, int pw)
    {
        int[] ODim = Cuda_conv3D.outputTensorDim(IH, IW, FH, FW, N, OC, sh, sw, ph, pw);
        int OH = ODim[1], OW = ODim[2];
        
        System.out.println("Test correct:");
        System.out.format("\tXsize( N, IC, IH, IW) = (%d, %d, %d, %d)\n", N, IC, IH, IW);
        System.out.format("\tWsize(OC, IC, FH, FW) = (%d, %d, %d, %d)\n", OC, IC, FH, FW);
        System.out.format("\tYsize( N, OC, OH, OW) = (%d, %d, %d, %d)\n", N, OC, OH, OW);
        
        int[] Gsize = Cuda_conv3D.getImg2colMatrixDim(OH, OW, FH, FW, N, IC, OC);
        int GN = Gsize[0], GM = Gsize[1], GK = Gsize[2];
        System.out.format("(GN, GM, GK) = (%d, %d, %d)\n", GN, GM, GK);
        
        int sizeX = N*IC*IH*IW;
	int sizeW = OC*IC*FH*FW;
	int sizeY = N*OC*OH*OW;
        
        float[] X = Vector.randomFloatVector(sizeX);
        float[] W = Vector.randomFloatVector(sizeW);
        
        //normal conv3D---------------------------------------------------------
        float[][][][] cX1 = Tense.vectorToTensor_4D(X, N, IH, IW, IC);
        float[][][][] cW1 = Tense.vectorToTensor_4D(W, OC, FH, FW, IC);
        float[][][][] cY1 = new float[N][OH][OW][OC];
        
        Tense.conv3D_naive(
                cX1, IH, IW,
                cW1, FH, FW,
                cY1, OH, OW,
                N, IC, OC, 
                sh, sw, ph, pw);
        
        float[] Y1 = Tense.tensor_4DToVector(cY1, sizeY);
        
        //fft conv3D------------------------------------------------------------
        float[][][][] cX2 = Tense.vectorToTensor_4D(X, N, IH, IW, IC);
        float[][][][] cW2 = Tense.vectorToTensor_4D(W, OC, FH, FW, IC);
        float[][][][] cY2 = new float[N][OH][OW][OC];
        
        FFT_v3.fft_conv3D_v1(
                cX2, IH, IW,
                cW2, FH, FW,
                cY2, OH, OW,
                N, IC, OC, 
                ph, pw);
        
        float[] Y2 = Tense.tensor_4DToVector(cY2, sizeY);
        
        //compare----------------------------------------------------------------
        float sp0 = Vector.samePercentRelative(Y1, Y2); 
        float zp0 = Vector.zeroPercent(Y1);
        float zp1 = Vector.zeroPercent(Y2);
        
        System.out.print("mmt-conv: "); Vector.println(Y1, 0, 10);
        System.out.print("fft-conv: "); Vector.println(Y2, 0, 10);
        System.out.println("sp0 = " + sp0);
        System.out.println("zp0 = " + zp0);
        System.out.println("zp1 = " + zp1);
        
        if(sp0 <0.999f) throw new RuntimeException();
    }
       
    public static void main(String[] args)
    {
        int IH = 1, IW = 8;
	int FH = 1, FW = 4;
	int N = 4;
	int IC = 16, OC = 16;
	int sh = 1, sw = 1, ph = 0, pw = 0;
        testCorrect(IH, IW, FH, FW, N, IC, OC, sh, sw, ph, pw);
    }
}
