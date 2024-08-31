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
public class Cuda_upool1D_avg_test 
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
        System.out.format("IW = %%d\n", IW);
        System.out.format("FW = %d\n", FW);
        System.out.format("OW = %d\n", OW);
        System.out.format("N, IC = %d, %d\n", N, IC);
        System.out.format("sw, pw = %d,, %d\n", sw, pw);
        System.out.format("GN, GM, GK = %d, %d, %d\n", GN, GM, GK);
        
        int sizeX = N * IW * IC;
        int sizeY = N * OW * IC;
        
        float[] deltaY = Vector.random_float_vector(sizeY, 1, 10);
        Tensor tdeltaY = eg.tensor(deltaY, N, OW, IC);
                
        //CPU-------------------------------------------------------------------
        float[][][][] cdeltaY = Vector.to4D(deltaY, N, 1, OW, IC);
        float[][][][] cdeltaX = new float[N][1][IW][IC];
        
//        Tense.unpool2D_avgerage_img2col_plus2(
//                cdeltaY, 1, OW,  1, FW,
//                cdeltaX, 1, IW,
//                N, IC, 
//                1, sw, 0, pw);
        
        Tense.unpool2D_avgerage_img2col_plus2_ignore_padding(
                cdeltaY, 1, OW, 1, FW,
                cdeltaX, 1, IW,
                N, IC, 
                1, sw, 0, pw);
       
        float[] deltaX1 = Tense.tensor_4DToVector(cdeltaX, sizeX);

        //GPU------------------------------------------------------------------
//        Tensor tdeltaX1 = eg.unpool1D_avg(false, tdeltaY, FW, sw, pw).c();
//        Tensor tdeltaX2 = eg.unpool1D_avg(false, eg.empty(N, IW, IC), tdeltaY, FW, sw).c();

        Tensor tdeltaX1 = eg.unpool1D_avg(true, tdeltaY, FW, sw, pw).c();
        Tensor tdeltaX2 = eg.unpool1D_avg(true, eg.empty(N, IW, IC).c(), tdeltaY, FW, sw).c();
        
        float[] deltaX2 = eg.valueOf(tdeltaX1);
        float[] deltaX3 = eg.valueOf(tdeltaX2);
        
        //compare---------------------------------------------------------------
        System.out.print("CPU1: "); Vector.println(deltaX1, 0, 10);
        System.out.print("GPU1: "); Vector.println(deltaX2, 0, 10);
        System.out.print("GPU2: "); Vector.println(deltaX3, 0, 10);
        
        float sp1 = Vector.samePercent_absolute(deltaX1, deltaX2); System.out.println("sp1: "+sp1);
        float sp2 = Vector.samePercent_absolute(deltaX1, deltaX3); System.out.println("sp3: "+sp2);
        float zp0 = Vector.zeroPercent(deltaX1); System.out.println("zp0: " + zp0);
        float zp1 = Vector.zeroPercent(deltaX2); System.out.println("zp1: " + zp1);
        
        eg.delete(tdeltaY, tdeltaX1, tdeltaX2);
        
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
