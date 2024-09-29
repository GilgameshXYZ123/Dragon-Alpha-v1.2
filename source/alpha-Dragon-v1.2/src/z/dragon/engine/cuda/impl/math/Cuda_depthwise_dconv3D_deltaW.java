/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.engine.cuda.impl.math;

import z.util.lang.annotation.Passed;

/**
 * deltaW[FH, FW, OC] = dconv3D(X[N, IH, IW, IC], deltaY[N, OH, OW, OC], [sh, sw], [ph, pw]).
 * X     [N, IH, IW, IC] -> [IC,     1, IH, IW, N]
 * deltaY[N, OH, OW, OC] -> [IC, OC/IC, OH, OW, N]
 * W = X * deltaY = [IC, OC/IC, FH, FW] -> [FH, FW, OC]
 * @author Gilgamesh
 */
public final class Cuda_depthwise_dconv3D_deltaW {
    private Cuda_depthwise_dconv3D_deltaW() {}
    
    //<editor-fold defaultstate="collapsed" desc="Common">
     public static int[] output_feature_dim(int IH, int IW, int FH, int FW, int N, int OC,
            int sh, int sw, int ph, int pw)
    {
        int OH = (IH + (ph << 1) - FH) / sh + 1;
        int OW = (IW + (pw << 1) - FW) / sw + 1;
        return new int[]{ N, OH, OW, OC };
    }
    
    public static int[] input_feature_dim(int OH, int OW, int FH, int FW, int N, int IC,
            int sh, int sw, int ph, int pw) 
    {
        int IH = (OH - 1)*sh + FH - 2*ph;
        int IW = (OW - 1)*sw + FW - 2*pw;
        return new int[]{ N, IH, IW, IC };
    }
    
    public static int[] im2col_matrix_dim(int FH, int FW, int OH, int OW, int N, int OC, int IC) {
        int GN = OC;
        int GM = FH * FW * IC;
        int GK = N  * OH * OW;
        return new int[]{GN, GM, GK};
    }
    
    public static int[] img2col_matrix_dim_logically(int OH, int OW, int FH, int FW, 
            int N, int OC, int IC, 
            int sh, int sw) 
    {
        int OHp = OH + (OH - 1) * (sh - 1);
        int OWp = OW + (OW - 1) * (sw - 1);
        int GN = OC;
        int GM = IC * FH * FW;
        int GK0 = N * OHp * OWp;
        return new int[]{GN, GM, GK0};
    }
    
    public static int[] out_padding(int ph, int pw) { return new int[]{ ph, pw }; }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="blockNum">
    public static final int GEMM_nblock(int FH, int FW, int OC, int LBX, int FD) {
        int GN = OC;
        int GM = FH * FW;
        int bx = (GN + (4 << LBX) - 1) >> 2 >> LBX;
        int by = (GM + FD - 1) / FD;
        return bx * by;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="class: GridZ_decider">
    public static interface DWConvDW_GZ_Decider {
        public int gridZ(int LBX, int LBK, int FD,
            int IH, int IW, int FH, int FW, int OH, int OW,
            int N, int IC, int OC);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="dconv3D_deltaW_GemmSK">
    /**
     * <pre>
     * deltaW[FH, FW, OC] = dconv3D(X[N, IH, IW, IC], deltaY[N, OH, OW, OC], [sh, sw], [ph, pw]).
     * </pre>
     * @param stream
     * @param GZ_decider
     * @param dX_address
     * @param IH
     * @param IW
     * @param d_deltaY_address
     * @param OH
     * @param OW
     * @param d_deltaW_address
     * @param FH
     * @param FW
     * @param N
     * @param IC
     * @param OC
     * @param sh
     * @param sw
     * @param ph
     * @param pw 
     */
    @Passed
    public native static void depthwise_dconv3D_deltaW_GemmSK(long stream, 
            DWConvDW_GZ_Decider GZ_decider,
            long       dX_address, int IH, int IW,
            long d_deltaY_address, int OH, int OW,
            long d_deltaW_address, int FH, int FW, 
            int N, int IC, int OC, 
            int sh, int sw, int ph, int pw);
    
    public static void Depthwise_dconv3D_deltaW_GemmSK(long stream, 
            int GridZ,
            long       dX_address, int IH, int IW,
            long d_deltaY_address, int OH, int OW,
            long d_deltaW_address, int FH, int FW, 
            int N, int IC, int OC, 
            int sh, int sw, int ph, int pw)
    {
        DWConvDW_GZ_Decider GZ = (LBX, LBK, FD, ih, iw, fh, fw, oh, ow, n, ic, oc) -> { return GridZ; };
        depthwise_dconv3D_deltaW_GemmSK(stream, GZ,
                dX_address, IH, IW, 
                d_deltaY_address, OH, OW, 
                d_deltaW_address, FH, FW,
                N, IC, OC,
                sh, sw, ph, pw);
    }
    //</editor-fold>
}
