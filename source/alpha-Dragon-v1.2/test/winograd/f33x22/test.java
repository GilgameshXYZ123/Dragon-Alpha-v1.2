/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package winograd.f33x22;

import winograd.f22x33.*;
import winograd.f2x3.*;
import static z.dragon.alpha.Alpha.alpha;
import z.dragon.engine.Engine;
import z.dragon.engine.memp.Mempool;
import z.util.math.vector.Tense;
import z.util.math.vector.Vector;

/**
 *
 * @author Gilgamesh
 */
public class test 
{
    public static void testCorrect(
            int IH, int IW, 
            int OH, int OW,
            int FH, int FW, 
            int N, int IC, int OC,
            int sh, int sw, int ph, int pw)
    {
        System.out.println("Test correct:");
        System.out.format("\tXsize( N, IC, IH, IW) = (%d, %d, %d, %d)\n", N, IC, IH, IW);
        System.out.format("\tWsize(OC, IC, FH, FW) = (%d, %d, %d, %d)\n", OC, IC, FH, FW);
        System.out.format("\tYsize( N, OC, OH, OW) = (%d, %d, %d, %d)\n", N, OC, OH, OW);
        
        int sizeX = N*IH*IW*IC;
	int sizeW = OC*IC*FH*FW;
	int sizeY = N*OC*OH*OW;
        
        float[] X = Vector.random_float_vector(sizeX);
        float[] W = Vector.random_float_vector(sizeW);
        
        //Direct----------------------------------------------------------------
        float[][][][] cX = Tense.vectorToTensor_4D(X, N, IH, IW, IC);
        float[][][][] cW = Tense.vectorToTensor_4D(W, OC, FH, FW, IC);
        float[][][][] cY1 = new float[N][OH][OW][OC];
        float[][][][] cY2 = new float[N][OH][OW][OC];
        
        Tense.conv3D_naive(
                cX, IH, IW,
                cW, FH, FW,
                cY1, OH, OW,
                N, IC, OC, 
                sh, sw, ph, pw);
        
        //Winograd--------------------------------------------------------------
        //Winograd2D_f33x22_v0.winograd(cX, IH, IW, cW, cY2, OH, OW, N, IC, OC, ph, pw);
        //Winograd2D_f33x22_v1.winograd(cX, IH, IW, cW, cY2, OH, OW, N, IC, OC, ph, pw);
        //Winograd2D_f33x22_v2.winograd(cX, IH, IW, cW, cY2, OH, OW, N, IC, OC, ph, pw);
        Winograd2D_f33x22_v3.winograd(cX, IH, IW, cW, cY2, OH, OW, N, IC, OC, ph, pw);
        
        //compare---------------------------------------------------------------
        float[] Y1 = Vector.flatten(cY1);
        float[] Y2 = Vector.flatten(cY2);
        float sp = Vector.samePercent_absolute(Y1, Y2);
        float zp0 = Vector.zeroPercent(Y1);
        float zp1 = Vector.zeroPercent(Y2);
        
        Vector.println("Y1 = ", Y1, 0, 10);
        Vector.println("Y2 = ", Y2, 0, 10);
        System.out.println("sp = " + sp);
        System.out.println("zp0 = " + zp0);
        System.out.println("zp1 = " + zp1);
        if(sp < 0.95f) throw new RuntimeException();
    }
    
    public static void main(String[] args)
    {
        int IH = 10, IW = 10;
	int OH = 8, OW = 8;
        
//        int IH = 18, IW = 18;
//	int OH = 16, OW = 16;
	int FH = 2, FW = 2;//4*4*8 = 32*4 = 128
        int IC = 32, OC = 32;//9*4=36 3*3*3 = 9*3 = 27;
	int N = 4;
	int sh = 1, sw = 1, ph = 0, pw = 0;
        
        for(int oc = 32; oc<=37; oc++)
            testCorrect(IH, IW, OH, OW, FH, FW, N, IC, oc, sh, sw, ph, pw);
    }
}
