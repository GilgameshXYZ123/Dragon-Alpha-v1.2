/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package test.cuda.pool1D;

import static z.dragon.alpha.Alpha.alpha;
import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.dragon.engine.cuda.impl.Cuda_expk2;

/**
 *
 * @author Gilgamesh
 */
public class Cuda_pool1D_indexed_test 
{
    static { alpha.home("C:\\Users\\Gilgamesh\\Desktop\\Dragon-alpha-v1.2");}
    static Engine eg = alpha.engine.cuda_float32(0, alpha.engine.memp1(alpha.MEM_1GB * 4));
     
    public static void testCorrect(
            int IW, int OW, int FW,
            int N, int IC, 
            int sw, int pw)
    {
        if(OW == -1) OW = (IW+pw*2 - FW)/sw+1;
        
        System.out.println("TestCorrect:");
        System.out.format("\t(IW) = (%d)\n", IW);
        System.out.format("\t(FW) = (%d)\n", FW);
        System.out.format("\t(OW) = (%d)\n", OW);
        System.out.format("\t(N, IC) = (%d, %d)\n", N, IC);
        System.out.format("\t(sw, pw) = (%d, %d)\n", sw, pw);
        
        int sizeX = N * IW * IC;
        int sizeY = N * OW * IC;
        
        //prepared--------------------------------------------------------------
        Tensor X      = eg.gaussian(0, 4, N, IW, IC).c();
        Tensor deltaY = eg.gaussian(0, 4, N, OW, IC).c();
        
        Tensor[] R = eg.pool1D_max_indexed(X, FW, sw, pw);
        Tensor Y = R[0].c(), Index = R[1];
        
        //GPU0------------------------------------------------------------------
        Tensor deltaX1 = eg.unpool1D_max(deltaY, Y, X, FW, sw, pw).c();
        Cuda_expk2.checkMemAlign(deltaX1);
        
        Tensor A1 = eg.reshape(false, deltaX1, N*IW, IC).c();
        Tensor B1 = eg.matMulT2(A1, A1).c();
        
        //GPU1------------------------------------------------------------------
        Tensor deltaX2 = eg.unpool1D_max_indexed(deltaY, Index, IW, FW, sw, pw).c();
        Cuda_expk2.checkMemAlign(deltaX2);
        Tensor A2 = eg.reshape(false, deltaX2, N*IW, IC).c();
        Tensor B2 = eg.matMulT2(A2, A2).c();
       
        //compare---------------------------------------------------------------
        float sp1 = eg.straight_equal(deltaX1, deltaX2).get();
        float sp2 = eg.straight_equal(B1, B2).get();
        
        System.out.println("sp1 = " + sp1);
        System.out.println("sp2 = " + sp2);
        
        if(sp1 < 0.99f) throw new RuntimeException();
        if(sp2 < 0.99f) throw new RuntimeException();
        
        eg.delete(A2, B2, A1, B1, deltaX1, deltaX2);
    }
    
    public static void main(String[] args) {
        {
            int IW = 64;
            int N = 16;
            int FW, sw, pw;
            
            FW = 4; sw = 2; pw = 1;
            for(int ic = 1; ic <= 128; ic++) testCorrect(IW, -1, FW, N, ic, sw, pw);
            
            FW = 3; sw = 2; pw = 1;
            for(int ic = 1; ic <= 128; ic++) testCorrect(IW, -1, FW, N, ic, sw, pw);
            
            FW = 2; sw = 2; pw = 0;
            for(int ic = 1; ic <= 128; ic++) testCorrect(IW, -1, FW, N, ic, sw, pw);
        }
    }
}
