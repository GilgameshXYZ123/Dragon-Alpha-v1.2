/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package test.cuda.dwconv3D;

import static z.dragon.alpha.Alpha.alpha;
import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.dragon.engine.cuda.CudaFloat32EngineBase;
import z.dragon.engine.cuda.impl.Cuda;
import z.util.lang.SimpleTimer;
import z.util.math.vector.Tense;
import z.util.math.vector.Vector;

/**
 *
 * @author Gilgamesh
 */
public class Cuda_dwconv3D_dW_test {
    static { alpha.home("C:\\Users\\Gilgamesh\\Desktop\\Dragon-alpha-v1.2");}
    static Engine eg = alpha.engine.cuda_float32(0, alpha.engine.memp1());
    static {
        CudaFloat32EngineBase cu32 = (CudaFloat32EngineBase) eg.engineBase();
    }
    
    public static void testCorrect(
            int IH, int IW,
            int OH, int OW, 
            int FH, int FW,
            int N, int IC, int OC,
            int sh, int sw, int ph, int pw)
    {
        if(IH == -1) IH = (OH - 1)*sh + FH - 2*ph;
        if(IW == -1) IW = (OW - 1)*sw + FW - 2*pw;
        
        int OHp = OH + (OH-1) * (sh-1), OWp = OW + (OW-1) * (sw-1);
        int oph = ph, opw = pw;
        
        System.out.println("\nTest correct:");
        System.out.format("\tYsize( N, OC, OH, OW) = (%d, %d, %d, %d)\n", N, OC, OH, OW);
        System.out.format("\tXsize( N, IC, IH, IW) = (%d, %d, %d, %d)\n", N, IC, IH, IW);
        System.out.format("\tWsize(OC, IC, FH, FW) = (%d, %d, %d, %d)\n", OC, IC, FH, FW);
        System.out.format("\t(sh, sw, ph, pw) = (%d, %d, %d, %d)\n", sh, sw, ph, pw);
        System.out.format("\t(OH_p, OW_p) = (%d, %d)\n", OHp, OWp);
        System.out.format("\t(oph, opw) = (%d, %d)\n", oph, opw);
        
        int GN = OC;
        int GM = IC * FH * FW;
        int GK0 = N * OHp * OWp;
        System.out.format("(GN, GM, GK0) = (%d, %d, %d)\n", GN, GM, GK0);
        
        int sizeX = N * IH * IW * IC;
        int sizeW = OC * FH * FW * IC;//sizeW_e = IC*OC(
        int sizeY = N * OH * OW * OC;
        
        float[] deltaY = Vector.random_float_vector(sizeY);
        float[] X = Vector.random_float_vector(sizeX);
        
        //CPU-------------------------------------------------------------------
        float[][][][] cY = Tense.vectorToTensor_4D(deltaY, N, OH, OW, OC);
        float[][][][] cX = Tense.vectorToTensor_4D(X, N, IH, IW, IC);
        float[][][][] cdeltaW = new float[OC][FH][FW][IC];
        
        Tense.deconv3d_deltaW_naive(cX, IH, IW, 
                cY, OH, OW,
                cdeltaW, FH, FW,
                N, IC, OC,
                sh, sw, ph, pw);
//        Tense.deconv3D_deltaW_img2col2(X1, IH, IW, 
//                deltaY1, OH, OW,
//                t_deltaW1, FH, FW,
//                N, IC, OC,
//                sh, sw, ph, pw);
                
        float[] deltaW1 = Tense.tensor_4DToVector(cdeltaW, sizeW); 
        System.out.print("CPU1: ");Vector.println(deltaW1, 0, 10);
        float zp0 = Vector.zeroPercent(deltaW1); 
        
        //GPU-------------------------------------------------------------------
        Tensor tX = eg.tensor(X, N, IH, IW, IC);
        Tensor tdeltaY = eg.tensor(deltaY, N, OH, OW, OC);
        Tensor tdeltaW1 = eg.conv3D_deltaW(tX, tdeltaY, FH, FW, sh, sw, ph, pw).c();
        Tensor tdeltaW2 = eg.conv3D_deltaW(eg.empty(OC, FH, FW, IC).c(), tX, tdeltaY, sh, sw).c();
        
        float[] deltaW2 = eg.valueOf(tdeltaW1);
        System.out.print("GPU1: "); Vector.println(deltaW2, 0, 10);
        float[] deltaW3 = eg.valueOf(tdeltaW2);
        System.out.print("GPU2: "); Vector.println(deltaW3, 0, 10);
        
        float zp3 = Vector.zeroPercent(deltaW2);
        
        //compare---------------------------------------------------------------
        float sp1 = Vector.samePercent_relative(deltaW1, deltaW2, 1e-3f); System.out.println("sp1: "+sp1);
        float sp2 = Vector.samePercent_relative(deltaW1, deltaW3, 1e-3f); System.out.println("sp2: "+sp2);
        
        System.out.println("zp0: " + zp0);
        System.out.println("zp3: "+zp3);
        
        if(sp1 != 1) {throw new RuntimeException();}
        if(sp2 != 1) {throw new RuntimeException();}
        
        eg.delete(tX, tdeltaY, tdeltaW1, tdeltaW2);
    }
    
    public static void testSpeed(int nIter,
            int IH, int IW,
            int OH, int OW, 
            int FH, int FW,
            int N, int IC, int OC,
            int sh, int sw, int ph, int pw)
    {
        if(IH == -1) IH = (OH - 1)*sh + FH - 2*ph;
        if(IW == -1) IW = (OW - 1)*sw + FW - 2*pw;

        int OHp = OH + (OH-1)*(sh-1), OWp = OW + (OW-1)*(sw-1);
        int oph = ph, opw = pw;
        
        System.out.println("Test speed:");
        System.out.format("\tYsize( N, OC, OH, OW) = (%d, %d, %d, %d)\n", N, OC, OH, OW);
        System.out.format("\tXsize( N, IC, IH, IW) = (%d, %d, %d, %d)\n", N, IC, IH, IW);
        System.out.format("\tWsize(OC, IC, FH, FW) = (%d, %d, %d, %d)\n", OC, IC, FH, FW);
        System.out.format("\t(sh, sw, ph, pw) = (%d, %d, %d, %d)\n", sh, sw, ph, pw);
        System.out.format("\t(OH_p, OW_p) = (%d, %d)\n", OHp, OWp);
        System.out.format("\t(oph, opw) = (%d, %d)\n", oph, opw);
        
        int GN = OC;
        int GM = FH * FW;
        int GK = N * OH * OW;
        int GK0 = N * OHp * OWp;
        System.out.format("(GN, GM, GK, GK0) = (%d, %d, %d, %d)\n", GN, GM, GK, GK0);
        
        int sizeX = N * IC * IH * IW;
        int sizeY = N * OC * OH * OW;
        
        float[] deltaY = Vector.random_float_vector(sizeY);
        float[] X = Vector.random_float_vector(sizeX);
        
        Tensor tX = eg.tensor(X, N, IH, IW, IC);
        Tensor tdeltaY = eg.tensor(deltaY, N, OH, OW, OC);
        Tensor tdeltaW = eg.empty(FH, FW, OC).c();
        
        tdeltaW = eg.depthwise_conv3D_deltaW(tdeltaW, tX, tdeltaY, sh, sw, ph, pw).c();
        //tdeltaW = eg.depthwise_conv3D_deltaW(tX, tdeltaY, FH, FW, sh, sw, ph, pw).c(); 
        System.out.println(tdeltaW.zero_percent());
        
        eg.check(false).sync(false);
        SimpleTimer timer = new SimpleTimer().record();
        for(int i = 0; i < nIter;i++) {
             tdeltaW = eg.depthwise_conv3D_deltaW(tdeltaW, tX, tdeltaY, sh, sw, ph, pw).c();
//            tdeltaW = eg.depthwise_conv3D_deltaW(tX, tdeltaY, FH, FW, sh, sw, ph, pw).c(); eg.delete(tdeltaW);
        }
        Cuda.deviceSynchronize();
        long div = timer.record().timeStamp_dif_millis();
        
        float time = 1.0f * div / nIter;
        float size  = 1.0f * GN / 1024 * GM / 1024 * GK / 1024;
        float total = 2 * 1024 * size * 1e-9f * 1024 * 1024;
        float performance = total / (time * 1e-3f);
        
        float size0 = 1.0f * GN / 1024 * GM / 1024 * GK0 / 1024;
        float total0 = 2 * 1024 * size0 * 1e-9f * 1024 * 1024;
	float performance0 = total0 / (time*1e-3f);
        
        System.out.format("Time = %f, Size = %f, Performance = %f GFlop/s, Size0 = %f, Performance0 = %f GFlop/s\n", 
                time, size, performance, size0, performance0);
        eg.delete(tX, tdeltaW, tdeltaY);
    }
    
    public static void main(String[] args) {
        //int FH = 2, FW = 2, ph = 1, pw = 1, sh = 1, sw = 1;
//        int FH = 3, FW = 3, ph = 1, pw = 1, sh = 1, sw = 1;
//	int FH = 4, FW = 4, ph = 2, pw = 2, sh = 1, sw = 1; 
//	int FH = 5, FW = 5, ph = 2, pw = 2, sh = 1, sw = 1;
//	int FH = 6, FW = 6, ph = 3, pw = 3, sh = 1, sw = 1;
	int FH = 7, FW = 7, ph = 3, pw = 3, sh = 1, sw = 1;
	//int FH = 8, FW = 8, ph = 4, pw = 4, sh = 1, sw = 1;
	//int FH = 9, FW = 9, ph = 4, pw = 4, sh = 1, sw = 1;
	//int FH = 10, FW = 10, ph = 5, pw = 5, sh = 1, sw = 1;
//	int FH = 11, FW = 11, ph = 5, pw = 5, sh = 1, sw = 1;
	//int FH = 12, FW = 12, ph = 6, pw = 6, sh = 1, sw = 1;
	//int FH = 13, FW = 13, ph = 6, pw = 6, sh = 1, sw = 1;
//	int FH = 14, FW = 14, ph = 7, pw = 7, sh = 1, sw = 1;
//	int FH = 15, FW = 15, ph = 7, pw = 7, sh = 1, sw = 1;
//	int FH = 16, FW = 16, ph = 8, pw = 8, sh = 1, sw = 1;

	//int IH = 128, IW = 128, OH = 128, OW = 128, N = 64, IC = 64, OC = 64;
	//int IH = 64, IW = 64, OH = 64, OW = 64, N = 64, IC = 128, OC = 128;
	int IH = 32, IW = 32, OH = 32, OW = 32, N = 128, IC = 256, OC = 256;
        
        testSpeed(500, IH, IW, OH, OW, FH, FW, N, IC, OC, sh, sw, ph, pw);
        testSpeed(500, IH - 1, IW - 1, OH - 1, OW - 1, FH, FW, N, IC, OC, sh, sw, ph, pw);
    }
    
}
