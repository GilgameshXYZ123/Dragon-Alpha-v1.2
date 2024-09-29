/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.engine.cuda.impl.math;

import z.util.lang.annotation.Passed;

/**
 * deltaX[N, IH, IW, IC] = depthwise_dconv3D(X[N, IH, IW, IC], W[FH, FW, OC], [sh, sw], [ph, pw]).
 * (1) deltaY[N, OH, OW, OC] -> deltaY[IC, N, OH, OW, OC/IC]
 * (2) W[FH, FW, OC]         -> W     [IC, 1, FH, FW, OC/IC] (rotate 180-degree)
 * (3) deltaY * W -> deltaX[IC, N, IH, IW, 1] -> deltaX[N, IH, IW, IC]
 * @author Gilgamesh
 */
public final class Cuda_depthwise_dconv3D_deltaX {
    private Cuda_depthwise_dconv3D_deltaX() {}
 
    //<editor-fold defaultstate="collapsed" desc="Common">
    public static int[] img2col_matrix_dim(int IH, int IW, int FH, int FW, int N, int IC, int OC)  {
        int GN = IC;
        int GM = N  * IH * IW;
        int GK = OC * FH * FW;
        return new int[]{ GN, GM, GK };
    }
    
    public static int[] output_feature_dim(int IH, int IW, int FH, int FW, int N, int OC,
            int sh, int sw, int ph, int pw)
    {
        int OH = (IH + (ph << 1) - FH) / sh + 1;
        int OW = (IW + (pw << 1) - FW) / sw + 1;
        return new int[]{N, OH, OW, OC};
    }
    
    public static int[] input_feature_dim(int OH, int OW, int FH, int FW, int N, int IC,
            int sh, int sw, int ph, int pw) 
    {
        int IH = (OH - 1)*sh + FH - 2*ph;
        int IW = (OW - 1)*sw + FW - 2*pw;
        return new int[]{ N, IH, IW, IC };
    }
    
    public static int[] out_paddding(int FH, int FW, int ph, int pw) {
        int oph = FH - 1 - ph;
        int opw = FW - 1 - pw;
        return new int[] { oph, opw};
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="blockNum">
    public static final int GEMM_nblock(int IH, int IW, int N, int IC) {
       int GN = IC;
       int GM = N * IH * IW;

       if ((GN > 31) && (GM > 127)) { return ((GN + 31) >> 5) * ((GM + 127) >> 7); }
       if ((GN > 63) && (GM >  63)) { return ((GN + 63) >> 6) * ((GM +  63) >> 6); }
       if ((GN > 31) && (GM >  63)) { return ((GN + 31) >> 5) * ((GM +  63) >> 6); }
       if ((GN > 15) && (GM >  63)) { return ((GN + 15) >> 4) * ((GM +  63) >> 6); }
       if ((GN > 15) && (GM >  31)) { return ((GN + 15) >> 4) * ((GM +  31) >> 5); }
       if ((GN > 15) && (GM >  15)) { return ((GN + 15) >> 4) * ((GM +  15) >> 4); }
       if ((GN >  7) && (GM >  31)) { return ((GN +  7) >> 3) * ((GM +  31) >> 5); }
       return                                ((GN +  7) >> 3) * ((GM +   7) >> 3);
    }
    //</editor-fold>
       
    //<editor-fold defaultstate="collapsed" desc="dconv3D_deltaX_s1">
    /**
     * <pre>
     * deltaX[N, IH, IW, IC] = depthwise_dconv3D(X[N, IH, IW, IC], W[FH, FW, OC], [sh, sw, ph, pw]).
     * </pre>
     * @param stream
     * @param d_deltaY_address
     * @param OH
     * @param OW
     * @param dW_address
     * @param FH
     * @param FW
     * @param d_deltaX_address
     * @param IH
     * @param IW
     * @param N
     * @param IC
     * @param OC
     * @param ph
     * @param pw 
     */
    @Passed
    public static native void depthwise_dconv3D_deltaX_s1(long stream,
            long d_deltaY_address, int OH, int OW,
            long       dW_address, int FH, int FW,
            long d_deltaX_address, int IH, int IW,
            int N, int IC, int OC,
            int ph, int pw);
    //</editor-fold>
}
