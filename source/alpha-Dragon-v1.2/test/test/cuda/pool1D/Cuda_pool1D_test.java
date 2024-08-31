package test.cuda.pool1D;

import static z.dragon.alpha.Alpha.alpha;
import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.dragon.engine.cuda.impl.CudaException;
import z.util.math.vector.Tense;
import z.util.math.vector.Vector;

/**
 *
 * @author Gilgamesh
 */
public class Cuda_pool1D_test 
{
    static { alpha.home("C:\\Users\\Gilgamesh\\Desktop\\Dragon-alpha-v1.2"); }
    static Engine eg = alpha.engine.cuda_float32(0, alpha.engine.memp1(alpha.MEM_1GB * 4), alpha.MEM_1MB * 2048);
     
    public static void testCorrect(
            int IW, int FW,
            int N, int IC, 
            int sw,int pw)
    {
        int OW = (IW + pw*2 - FW) / sw+1;
        
        int GN = IC;
        int GM = N * OW;
        int GK = FW;
        
        System.out.println("TestCorrect:");
        System.out.format("\t(IW) = (%d)\n", IW);
        System.out.format("\t(FW) = (%d)\n", FW);
        System.out.format("\t(OW) = (%d)\n", OW);
        System.out.format("\t(N, IC) = (%d, %d)\n", N, IC);
        System.out.format("\t(sw, pw) = (%d, %d)\n", sw, pw);
        System.out.format("\t(GN, GM, GK) = (%d, %d, %d)\n", GN, GM, GK);
        
        int sizeX = N * IW * IC;
        int sizeY = N * OW * IC;
        
        float[] X = Vector.random_float_vector(sizeX, -19, 19);
        Tensor tX = eg.tensor(X, N, IW, IC);
        
        //CPU-------------------------------------------------------------------
        float[][][] cX = Vector.to3D(X, N, IW, IC);
        float[][][] cY = new float[N][OW][IC];
        
        Tense.pool1D_avg_naive_ignore_padding(cX, IW, FW, cY, OW, N, IC, sw, pw);
//        Tense.pool1D_avg_naive(cX, IW, FW, cY, OW, N, IC, sw, pw);
//        Tense.pool1D_max_naive(cX, IW, FW, cY, OW, N, IC, sw, pw);
        
        float[] Y1 = Vector.flatten(cY); 
        
        //GPU-------------------------------------------------------------------
        Tensor tY1 = eg.pool1D_avg(true, tX, FW, sw, pw).c();
        Tensor tY2 = eg.pool1D_avg(true, eg.empty(N, OW, IC).c(), tX, FW, sw, pw).c();
        
//        Tensor tY1 = eg.pool1D_avg(false, tX, FW, sw, pw).c();
//        Tensor tY2 = eg.pool1D_avg(false, eg.empty(N, OW, IC).c(), tX, FW, sw, pw).c();

//        Tensor tY1 = eg.pool1D_max(tX, FW, sw, pw).c();
//        Tensor tY2 = eg.pool1D_max(eg.empty(N, OW, IC).c(), tX, FW, sw, pw).c();
        
        System.out.println(CudaException.lastException());
        
        float[] Y2 = eg.valueOf(tY1);
        float[] Y3 = eg.valueOf(tY2);

        //----------------------------------------------------------------------
        float zp0 = Vector.zeroPercent(Y1); System.out.println("zp0: " + zp0);
        float zp1 = Vector.zeroPercent(Y2); System.out.println("zp1 = " + zp1);
        float zp2 = Vector.zeroPercent(Y3); System.out.println("zp2 = " + zp2);
        
        System.out.print("CPU: ");  Vector.println(Y1, 0, 10);
        System.out.print("GPU1: "); Vector.println(Y2, 0, 10);
        System.out.print("GPU2: "); Vector.println(Y3, 0, 10);
        
        //compare---------------------------------------------------------------
        float sp1 = Vector.samePercent_absolute(Y1, Y2); System.out.println("sp: " + sp1);
        float sp2 = Vector.samePercent_absolute(Y1, Y3); System.out.println("sp: " + sp2);
        
        if(sp1 != 1) System.out.println("12312313123");
        if(sp1 != 1) throw new RuntimeException("IC = " + IC);
        
        eg.delete(tX, tY1, tY2);
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
