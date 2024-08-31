package test.cuda.pool1D;

import static z.dragon.alpha.Alpha.alpha;
import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.util.math.vector.Tense;
import z.util.math.vector.Vector;

/**
 *
 * @author Gilgamesh
 */
public class Cuda_upool1D_max_test 
{
    static { alpha.home("C:\\Users\\Gilgamesh\\Desktop\\Dragon-alpha-v1.2");}
    static Engine eg = alpha.engine.cuda_float32(0, alpha.engine.memp1());
    
    public static void testCorrect(
            int OW, int FW, int N, int IC,
            int sw, int pw)
    {
        int IW = (OW - 1)*sw + FW - 2*pw;
        
        int GN = IC;
        int GM = N * IW;
        int GK = FW;
        
        System.out.println("Test Correct:");
        System.out.format("IW = %d\n", IW);
        System.out.format("FW = %d\n", FW);
        System.out.format("OW = %d\n", OW);
        System.out.format("N, IC = %d, %d\n", N, IC);
        System.out.format("sw, pw = %d, %d\n", sw, pw);
        System.out.format("GN, GM, GK = %d, %d, %d\n", GN, GM, GK);
        
        int sizeX = N * IW * IC;
        int sizeY = N * OW * IC;
        
        float[] deltaY = Vector.random_float_vector(sizeY);
        float[] Y = Vector.random_float_vector(sizeY);
        float[] X = Vector.random_float_vector(sizeX);
        
        //CPU-------------------------------------------------------------------
        float[][][][] cdeltaY = Vector.to4D(deltaY, N, 1, OW, IC);
        float[][][][] cY      = Vector.to4D(Y, N, 1, OW, IC);
        float[][][][] cX      = Vector.to4D(X, N, 1, IW, IC);
                
        Tensor tdeltaY =  eg.tensor(deltaY, N, OW, IC).c();
        Tensor tY = eg.tensor(Y, N, OW, IC).c();
        Tensor tX = eg.tensor(X, N, IW, IC).c();
        
        //CPU-------------------------------------------------------------------
        float[][][][] cdeltaX = new float[N][1][IW][IC];
        Tense.unpool2D_max_img2col_plus2(
                cdeltaY, cY, 1, OW, 1, FW,
                cdeltaX, cX, 1, IW,
                N, IC, 
                1, sw, 0, pw);
       
        float[] deltaX1 = Vector.flatten(cdeltaX);
        System.out.print("CPU1: "); Vector.println(deltaX1, 0, 10);

        //GPU------------------------------------------------------------------
        Tensor tdeltaX1 = eg.unpool1D_max(eg.empty(N, IW, IC).c(), tdeltaY, tY, tX, FW, sw).c();
        Tensor tdeltaX2 = eg.unpool1D_max(tdeltaY, tY, tX, FW, sw, pw).c();
        
        float[] deltaX2 = eg.valueOf(tdeltaX1);
        System.out.print("GPU : "); Vector.println(deltaX2, 0, 10);
        
        float[] deltaX3 = eg.valueOf(tdeltaX2);
        System.out.print("GPU : "); Vector.println(deltaX3, 0, 10);
        
        float sp1 = Vector.samePercent_absolute(deltaX1, deltaX2); System.out.println("sp1: "+sp1);
        float sp2 = Vector.samePercent_absolute(deltaX1, deltaX3); System.out.println("sp2: "+sp2);
        
        float zp0 = Vector.zeroPercent(deltaX1); System.out.println("zp0: " + zp0);
        float zp1 = Vector.zeroPercent(deltaX2); System.out.println("zp1: " + zp1);
        
        eg.delete(tdeltaY, tY, tX, tdeltaX1, tdeltaX2);
        
        if(sp1!=1) throw new RuntimeException();
    }
    
    public static void main(String[] args) { 
        {
            int IW = 64;
            int N = 16;
            int FW, sw, pw;
            
            FW = 4; sw = 2; pw = 1;
            for(int ic = 1; ic <= 128; ic++) testCorrect(IW, FW, N, ic, sw, pw);
            
            FW = 3; sw = 2; pw = 1;
            for(int ic = 1; ic <= 128; ic++) testCorrect(IW, FW, N, ic, sw, pw);
            
            FW = 2; sw = 2; pw = 0;
            for(int ic = 1; ic <= 128; ic++) testCorrect(IW, FW, N, ic, sw, pw);
        }
    }
}
