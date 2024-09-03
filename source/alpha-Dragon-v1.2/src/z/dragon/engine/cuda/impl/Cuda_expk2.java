/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.engine.cuda.impl;

import z.dragon.engine.Tensor;
import z.util.lang.annotation.Passed;

/**
 *
 * @author Gilgamesh
 */
public final class Cuda_expk2 {
    private Cuda_expk2() {}
    
    //<editor-fold defaultstate="collapsed" desc="Tenosr Transpose(2D -> 4D)">
    /**
     * <pre>
     * transpose2D: X[dim0, dim1] -> Y[dim1, dim0].
     * (1) mem_strideX = (Xdim1 + 3) / 4 * 4
     * (2) mem_strideY = (Ydim1 + 3) / 4 * 4
     * (3) Xdim1 = Ydim0, Xdim0 = Ydim1
     * </pre>
     *
     * @param cudaStream_address
     * @param dX_address
     * @param dY_address
     * @param Xdim1
     * @param Ydim1
     * @param mem_strideX
     * @param mem_strideY
     * @param length
     */
    public static native void transpose2D(long cudaStream_address,
            long dX_address, long dY_address,
            int Xdim1, int Ydim1,
            int mem_strideX, int mem_strideY,
            int length);

    /**
     * <pre>
     * transpose3D; X[Xdim0, Xdim1, Xdim2] -> Y[Ydim0, Ydim1, Ydim].
     * (1) mem_strideX = (Xdim1 + 3) / 4 * 4
     * (2) mem_strideY = (Ydim1 + 3) / 4 * 4
     * (3) mul(Xdim) = mul(Ydim) = length
     * </pre>
     *
     * @param cudaStream_address
     * @param dX_address
     * @param dY_address
     * @param Xdim1
     * @param Xdim2
     * @param Ydim1
     * @param Ydim2
     * @param dimIndex1
     * @param dimIndex2
     * @param mem_strideX
     * @param mem_strideY
     * @param length
     */
    public static native void transpose3D(long cudaStream_address,
            long dX_address, long dY_address,
            int Xdim1, int Xdim2,
            int Ydim1, int Ydim2,
            int dimIndex1, int dimIndex2,
            int mem_strideX, int mem_strideY,
            int length);

    /**
     * <pre>
     * transpose4D; X[Xdim0, Xdim1, Xdim2, Xdim3] -> Y[Ydim0, Ydim1, Ydim, Ydim3].
     * (1) mem_strideX = (Xdim1 + 3) / 4 * 4
     * (2) mem_strideY = (Ydim1 + 3) / 4 * 4
     * (3) mul(Xdim) = mul(Ydim) = length
     * </pre>
     *
     * @param cudaStream_address
     * @param dX_address
     * @param dY_address
     * @param Xdim1
     * @param Xdim2
     * @param Xdim3
     * @param Ydim1
     * @param Ydim2
     * @param Ydim3
     * @param dimIndex1
     * @param dimIndex2
     * @param mem_strideX
     * @param mem_strideY
     * @param length
     */
    public static native void transpose4D(long cudaStream_address,
            long dX_address, long dY_address,
            int Xdim1, int Xdim2, int Xdim3,
            int Ydim1, int Ydim2, int Ydim3,
            int dimIndex1, int dimIndex2,
            int mem_strideX, int mem_strideY,
            int length);
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Gapped Memcpy 2D">
    /**
     * <pre>
     * |[width][gapX]|: width + gapX = strideX
     * |[width][gapY]|: width + gapY = strideY.
     * (1) (strideX, strideY) % 4 == 0
     * (2) (strideX, strideY) >= width
     * </pre>
     *
     * @param stream_address
     * @param dX_address
     * @param Xstart
     * @param strideX
     * @param dY_address
     * @param Ystart
     * @param strideY
     * @param width
     * @param length
     */
    @Passed
    public static native void gappedMemcpy2D(long stream_address,
            long dX_address, int Xstart, int strideX,
            long dY_address, int Ystart, int strideY,
            int width, int length);
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Rotate 180">
    /**
     * <pre>
     * X[N, IH, IW, IC] -> Xr[N, IH, IW, IC]
     * Xr(n, IH - 1- ih, IW - 1 - iw, ic) = X(n, ih, iw, ic).
     * (1) IC % 4 == 0
     * (2) N % 4 != 0
     * (3) length = N*IH*IW*IC %4 == 0
     * </pre>
     *
     * @param cudaStream_address
     * @param dX_address
     * @param dY_address
     * @param IH
     * @param IW
     * @param IC
     * @param length
     */
    public static native void rot180(long cudaStream_address,
            long dX_address, long dY_address,
            int IH, int IW, int IC,
            int length);
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="pool indexed">
    /**
     * <pre>
     * the src Tensor is Indexed:
     *      for i = 0:lengthv: Y[i] = X[Index[i]].
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     * (4) linear(beta) != 0
     * </pre>
     * @param cudaStream_address
     * @param dX_address
     * @param dIndex_address
     * @param dY_address
     * @param lengthv 
     * @param width 
     * @param stride 
     */
    public static native void srcIndexedMemcpy(long cudaStream_address,
            long dX_address, long dIndex_address,
            long dY_address,
            int lengthv, int width, int stride); 
    
    /**
     * <pre>
     * the dst Tensor is Indexed:
     *      for i = 0: lengthv: Y[Index[i]] = X[i].
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     * (4) linear(beta) != 0
     * </pre>
     * @param cudaStream_address
     * @param dX_address
     * @param dIndex_address
     * @param dY_address
     * @param lengthv 
     * @param width 
     * @param stride 
     */
    public static native void dstIndexedMemcpy(long cudaStream_address,
            long dX_address, long dIndex_address,
            long dY_address,
            int lengthv, int width, int stride); 
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="pad(2D -> 4D)">
    /**
     * <pre>
     * X[IN, IC] -> Y[ON, IC]
     * (1) IN -> ON = pn0 + IN + pn1
     * (2) IC -> OC = pc0 + IC + pc1.
     * [1] (IN, ON) % 4 != 0 (ignore the mem-alginment on N)
     * [2] (IC, OC) % 4 == 0
     * </pre>
     *
     * @param cudaStream_address
     * @param dX_address
     * @param IN
     * @param IC
     * @param dY_address
     * @param ON
     * @param OC
     * @param pn0
     * @param pc0
     */
    public static native void pad2D(long cudaStream_address,
            long dX_address, int IN, int IC,
            long dY_address, int ON, int OC,
            int pn0, int pc0);

    /**
     * <pre>
     * X[IN, IW, IC] -> Y[ON, OW, OC]
     * (1) IN -> ON = pn0 + IN + pn1
     * (2) IW -> OW = pw0 + IW + pw1
     * (3) IC -> OC = pc0 + IC + pc1.
     * [1] (IN, ON) % 4 != 0 (ignore the mem-alginment on N)
     * [2] (IC, OC) % 4 == 0
     * </pre>
     *
     * @param cudaStream_address
     * @param dX_adresss
     * @param IN
     * @param IW
     * @param IC
     * @param dY_address
     * @param ON
     * @param OW
     * @param OC
     * @param pn0
     * @param pw0
     * @param pc0
     */
    public static native void pad3D(long cudaStream_address,
            long dX_adresss, int IN, int IW, int IC,
            long dY_address, int ON, int OW, int OC,
            int pn0, int pw0, int pc0);

    /**
     * <pre>
     * X[IN, IH, IW, IC] -> Y[ON, OH, OW, OC]
     * (1) IN -> ON = pn0 + IN + pn1
     * (2) IH -> OH = ph0 + IH + ph1
     * (3) IW -> OW = pw0 + IW + pw1
     * (4) IC -> OC = pc0 + IC + pc1.
     * [1] (IN, ON) % 4 != 0 (ignore the mem-alginment on N)
     * [2] (IC, OC) % 4 == 0
     * </pre>
     *
     * @param cudaStream_address
     * @param dX_address
     * @param IN
     * @param IH
     * @param IW
     * @param IC
     * @param dY_address
     * @param ON
     * @param OH
     * @param OW
     * @param OC
     * @param pn0
     * @param ph0
     * @param pw0
     * @param pc0
     */
    public static native void pad4D(long cudaStream_address,
            long dX_address, int IN, int IH, int IW, int IC,
            long dY_address, int ON, int OH, int OW, int OC,
            int pn0, int ph0, int pw0, int pc0);
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="trim(2D -> 4D)">
    /**
     * <pre>
     * X[IN, IC] -> Y[ON, IC]
     * (1) IN = pn0 + ON + pn1 -> ON
     * (2) IC = pc0 + OC + pc1 -> OC.
     * [1] (IN, ON) % 4 != 0 (ignore the mem-alginment on N)
     * [2] (IC, OC) % 4 == 0
     * </pre>
     *
     * @param cudaStream_address
     * @param dX_address
     * @param IN
     * @param IC
     * @param dY_address
     * @param ON
     * @param OC
     * @param pn0
     * @param pc0
     */
    public static native void trim2D(long cudaStream_address,
            long dX_address, int IN, int IC,
            long dY_address, int ON, int OC,
            int pn0, int pc0);

    /**
     * <pre>
     * X[IN, IW, IC] -> Y[ON, OW, OC]
     * (1) IN = pn0 + ON + pn1 -> ON
     * (2) IW = pw0 + OW + pw1 -> OW
     * (3) IC = pc0 + OC + pc1 -> OC.
     * [1] (IN, ON) % 4 != 0 (ignore the mem-alginment on N)
     * [2] (IC, OC) % 4 == 0
     * </pre>
     *
     * @param cudaStream_address
     * @param dX_adresss
     * @param IN
     * @param IW
     * @param IC
     * @param dY_address
     * @param ON
     * @param OW
     * @param OC
     * @param pn0
     * @param pw0
     * @param pc0
     */
    public static native void trim3D(long cudaStream_address,
            long dX_adresss, int IN, int IW, int IC,
            long dY_address, int ON, int OW, int OC,
            int pn0, int pw0, int pc0);

    /**
     * <pre>
     * X[IN, IH, IW, IC] -> Y[ON, OH, OW, OC]
     * (1) IN = pn0 + ON + pn1 -> ON
     * (2) IH = ph0 + OH + ph1 -> OH
     * (3) IW = pw0 + OW + pw1 -> OW
     * (4) IC = pc0 + OC + pc1 -> OC.
     * [1] (IN, ON) % 4 != 0 (ignore the mem-alginment on N)
     * [2] (IC, OC) % 4 == 0
     * </pre>
     *
     * @param cudaStream_address
     * @param dX_address
     * @param IN
     * @param IH
     * @param IW
     * @param IC
     * @param dY_address
     * @param ON
     * @param OH
     * @param OW
     * @param OC
     * @param pn0
     * @param ph0
     * @param pw0
     * @param pc0
     */
    public static native void trim4D(long cudaStream_address,
            long dX_address, int IN, int IH, int IW, int IC,
            long dY_address, int ON, int OH, int OW, int OC,
            int pn0, int ph0, int pw0, int pc0);
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="check mem alignment">
    public static void checkMemAlign(Tensor X) {
        int width = X.lastDim();
        int stride = ((width + 3) >> 2) << 2;
        System.out.println("check_mem_align");
        check_mem_alignment(0, X.address(), X.lengthv(), width, stride);
    }

    /**
     * <pre>
     * check the correctless of memAlignment.
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     * (4) linear(beta) != 0
     * </pre>
     *
     * @param cudaStream_address
     * @param dX_address
     * @param lengthv
     * @param width
     * @param stride
     */
    public static native void check_mem_alignment(long cudaStream_address,
            long dX_address,
            int lengthv, int width, int stride);
    //</editor-fold>
}
