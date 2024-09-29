/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.engine.cuda.impl.math;

/**
 * Y[N, OH, OW, OC] = depthwise_conv3D(X[N, IH, IW, IC], W[FH, FW, OC], [ph, pw, sh, sw]).
 * (1) X[N, IH, IW, IC] -> X[IC,     N, IH, IW, 1]
 * (2) W[FH, FW, OC]    -> W[IC, OC/IC, FH, FW, 1]
 * (3) X * W -> Y[IC, N, OH, OW, OC/IC] -> Y[N, OH, OW, OC]
 * @author Gilgamesh
 */
public final class Cuda_depthwise_conv3D {
    private Cuda_depthwise_conv3D() {}
    
    //<editor-fold defaultstate="collapsed" desc="Common">
    public static int[] img2col_matrix_dim(int OH, int OW, int FH, int FW, int N, int OC) {
        int GN = OC;
        int GM = N * OH * OW;
        int GK = FH * FW;
        return new int[]{ GN, GM, GK };
    }
    
    public static int[] output_feature_dim(int IH, int IW, int FH, int FW,  int N, int OC,
            int sh, int sw, int ph, int pw) 
    {
        int OH = (IH + (ph << 1) - FH) / sh + 1;
        int OW = (IW + (pw << 1) - FW) / sw + 1;
        return new int[]{ N, OH, OW, OC };
    }
    
    public static int[] input_feature_size(int OH, int OW, int FH, int FW, int N, int IC,
            int sh, int sw, int ph, int pw) 
    {
        int IH = (OH - 1)*sh + FH - 2*ph;
        int IW = (OW - 1)*sw + FW - 2*pw;
        return new int[]{ N, IH, IW, IC };
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="blockNum">
    public static final int GEMM_nblock(int OH, int OW, int N, int OC) {
       int GN = OC;
       int GM = N * OH * OW;
       
       if ((GN > 63) && (GM >  63)) { return ((GN + 63) >> 6) * ((GM +  63) >> 6); }
       if ((GN > 31) && (GM > 127)) { return ((GN + 31) >> 5) * ((GM + 127) >> 7); }
       if ((GN > 31) && (GM >  63)) { return ((GN + 31) >> 5) * ((GM +  63) >> 6); }
       if ((GN > 15) && (GM >  63)) { return ((GN + 15) >> 4) * ((GM +  63) >> 6); }
       if ((GN > 15) && (GM >  31)) { return ((GN + 15) >> 4) * ((GM +  31) >> 5); }
       if ((GN > 15) && (GM >  15)) { return ((GN + 15) >> 4) * ((GM +  15) >> 4); }
       if ((GN >  7) && (GM >  31)) { return ((GN +  7) >> 3) * ((GM +  31) >> 5); }
       return                                ((GN +  7) >> 3) * ((GM +   7) >> 3);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="depthwise_conv3D">
    /**
     * Y[N, OH, OW, OC] = depthwise_conv3D(X[N, IH, IW, IC], W[FH, FW, OC], [ph, pw, sh, sw]).
     * (1) GK = FH * FW >= 1
     * (2) GM % 4 ==0, GM >= 4
     * @param stream
     * @param dX_address
     * @param IH
     * @param IW
     * @param dW_address
     * @param FH
     * @param FW
     * @param dY_address
     * @param OH
     * @param OW
     * @param N
     * @param IC
     * @param OC
     * @param sh
     * @param sw
     * @param ph
     * @param pw 
     */
    public static native void depthwise_conv3D(long stream,
            long dX_address, int IH, int IW,
            long dW_address, int FH, int FW,
            long dY_address, int OH, int OW,
            int N, int IC, int OC,
            int sh, int sw, int ph, int pw);
    //</editor-fold>
}
