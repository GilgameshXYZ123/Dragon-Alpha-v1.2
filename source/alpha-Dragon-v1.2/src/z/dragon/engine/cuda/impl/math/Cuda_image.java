/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.engine.cuda.impl.math;

/**
 *
 * @author Gilgamesh
 */
public final class Cuda_image {
    private Cuda_image() {}
    
    //<editor-fold defaultstate="collapsed" desc="image: char2float">
    //<editor-fold defaultstate="collapsed" desc="linear2D: pixel2float">
    /**
     * <pre>
     * Linear Transformation: Y(float) = alpha * X(uint8) + beta.
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     * (4) linear(beta) != 0
     *
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * [height, width] = [1024, 1024]: Time = 0.108000, Speed = 72.337959 GB/s
     * </pre>
     *
     * @param cudaStream_address
     * @param alpha
     * @param dX_address
     * @param beta
     * @param dY_address
     * @param lengthv
     * @param mem_width
     * @param mem_stride
     */
    public static native void linear2D_pixel2float(long cudaStream_address,
            float alpha, long dX_address, float beta,
            long dY_address,
            int lengthv, int mem_width, int mem_stride);

    /**
     * <pre>
     * Linear Transformation: Y(uint8) = alpha * X(float) + beta.
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     * (4) linear(beta) != 0
     *
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * [height, width] = [1024, 1024]: Time = 0.108000, Speed = 72.337959 GB/s
     * </pre>
     *
     * @param cudaStream_address
     * @param alpha
     * @param dX_address
     * @param beta
     * @param dY_address
     * @param lengthv
     * @param mem_width
     * @param mem_stride
     */
    public static native void linear2D_float2pixel(long cudaStream_address,
            float alpha, long dX_address, float beta,
            long dY_address,
            int lengthv, int mem_width, int mem_stride);
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="image: img_dualLinear2_div2D">
    /**
     * <pre>
     *  [1] Y1 = alpha1*X(uint8) + beta1*X1(uint8) + gamma1
     *  [2] Y2 = alpha2*X(uint8) + beta2*X2(uint8) + gamma2
     *  [3] Y(float) = (Y1 / Y2) + C.
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     * </pre>
     *
     * @param stream_address
     * @param dX_address
     * @param dX1_address
     * @param dX2_address
     * @param alpha1
     * @param beta1
     * @param gamma1
     * @param alpha2
     * @param beta2
     * @param gamma2
     * @param C
     * @param dY_address
     * @param lengthv
     * @param mem_width
     * @param mem_stride
     */
    public static native void img_dualLinear2_div2D(long stream_address,
            long dX_address,
            long dX1_address,
            long dX2_address,
            float alpha1, float beta1, float gamma1,
            float alpha2, float beta2, float gamma2, float C,
            long dY_address,
            int lengthv, int mem_width, int mem_stride);
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="image: img_dualLinear2_normalize2D: row, center">
    /**
     * <pre>
     *  [1] Y1 = alpha1*X(uint8) + beta1*X1(float) + gamma1
     *  [2] Y2 = alpha2*X(float) + beta2*X2(float) + gamma2
     *  [3] Y(float) = (Y1 / Y2) + C
     *  [4] X[field, row], X1[row], X2[row], Y[field, row].
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     * </pre>
     *
     * @param stream_address
     * @param dX_address
     * @param dX1_address
     * @param dX2_address
     * @param row_lengthv
     * @param alpha1
     * @param beta1
     * @param gamma1
     * @param alpha2
     * @param beta2
     * @param gamma2
     * @param C
     * @param dY_address
     * @param lengthv
     * @param mem_width
     * @param mem_stride
     */
    public static native void img_dualLinear2_noramlize2D_row(long stream_address,
            long dX_address,
            long dX1_address,
            long dX2_address, int row_lengthv,
            float alpha1, float beta1, float gamma1,
            float alpha2, float beta2, float gamma2, float C,
            long dY_address,
            int lengthv, int mem_width, int mem_stride);

    /**
     * <pre>
     *  [1] Y1 = alpha1*X(uint8) + beta1*X1(float) + gamma1
     *  [2] Y2 = alpha2*X(float) + beta2*X2(float) + gamma2
     *  [3] Y(float) = (Y1 / Y2) + C
     *  [4] X[dim0, dim1, dim2], X1[dim0, dim2], X2[dim0 ,dim2], Y[dim0, dim1, dim2].
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     * </pre>
     *
     * @param stream_address
     * @param dX_address
     * @param dX1_address
     * @param dX2_address
     * @param alpha1
     * @param beta1
     * @param gamma1
     * @param alpha2
     * @param beta2
     * @param gamma2
     * @param C
     * @param dim0
     * @param dY_address
     * @param dim2
     * @param dim1
     * @param mem_width
     * @param mem_stride
     */
    public static native void img_dualLinear2_noramlize2D_center(long stream_address,
            long dX_address,
            long dX1_address,
            long dX2_address,
            float alpha1, float beta1, float gamma1,
            float alpha2, float beta2, float gamma2, float C,
            long dY_address, int dim0, int dim1, int dim2,
            int mem_width, int mem_stride);
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="image: linear2_div2D: field, row">
    /**
     * <pre>
     * (1) X1.width = X2.width = Y.width
     * (2) X1.[length, lengthv] = Y.[length, lengthv] = [length, lengthv]
     * (3) [length, lengthv] % X2.[length, lengthv] == 0
     * (4) reshape: Xrow2 -> Xrow2[Xrow2.length]
     * (5) reshape: X1, Y -> (X1, Y)[length/X2.length, X2.length]
     * (6) X1[i], Y[i] is the ith row vector of X1, Y:
     *      for i from 1 to X2_length:
     *          Y1 = alpha1*X[i](uint8) + beta1*X2(float) + gamma
     *          Y2 = alpha2*X2(float) + beta2
     *          Y[i](float) = Y1 / Y2 + C.
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     *
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * [height, width] = [1024, 1024]: Time = 0.110000, Speed = 106.534081 GB/s
     * </pre>
     *
     * @param cudaStream_address
     * @param dX_address
     * @param dX1_address
     * @param dX2_address
     * @param row_lengthv
     * @param alpha1
     * @param beta1
     * @param gamma1
     * @param alpha2
     * @param beta2
     * @param C
     * @param dY_address
     * @param lengthv
     * @param width
     * @param stride
     */
    public static native void img_linear2_div2D_field(long cudaStream_address,
            long dX_address,
            long dX1_address,
            long dX2_address, int row_lengthv,
            float alpha1, float beta1, float gamma1,
            float alpha2, float beta2, float C,
            long dY_address,
            int lengthv, int width, int stride);

    /**
     * <pre>
     * (1) X1.width = X2.width = Y.width
     * (2) X1.[length, lengthv] = Y.[length, lengthv] = [length, lengthv]
     * (3) [length, lengthv] % X2.[length, lengthv] == 0
     * (4) reshape: Xrow2 -> Xrow2[Xrow2.length]
     * (5) reshape: X1, Y -> (X1, Y)[length/X2.length, X2.length]
     * (6) X1[i], Y[i] is the ith row vector of X1, Y:
     *      for i from 1 to X2_length:
     *          Y1 = alpha1*X[i](int8) + beta1*X2(float) + gamma
     *          Y2 = alpha2*X2(float) + beta2
     *          Y[i](float) = Y1 / Y2 + C.
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     *
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * [height, width] = [1024, 1024]: Time = 0.110000, Speed = 106.534081 GB/s
     * </pre>
     *
     * @param cudaStream_address
     * @param dX_address
     * @param dX1_address
     * @param dX2_address
     * @param row_lengthv
     * @param alpha1
     * @param dY_address
     * @param beta1
     * @param alpha2
     * @param beta2
     * @param C
     * @param lengthv
     * @param gamma1
     * @param mem_width
     * @param mem_stride
     */
    public static native void img_linear2_div2D_row(long cudaStream_address,
            long dX_address,
            long dX1_address,
            long dX2_address, int row_lengthv,//row_lengthv = X2_lengthv
            float alpha1, float beta1, float gamma1,
            float alpha2, float beta2, float C,
            long dY_address,
            int lengthv, int mem_width, int mem_stride);
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="image: linear, exp, log">
    //<editor-fold defaultstate="collapsed" desc="image: linear, linear2D_dual_row, field">
    /**
     * <pre>
     * Linear Transformation:
     *  [1] Y = alpha * X + beta
     *  [2] Y = clip(Y, 0, 255).
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     * (4) the datatype of Y&X is int8
     *
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * [height, width] = [1024, 1024]: Time = 0.108000, Speed = 72.337959 GB/s
     * </pre>
     *
     * @param stream_address
     * @param alpha
     * @param dX_address
     * @param beta
     * @param dY_address
     * @param lengthv
     * @param mem_width
     * @param mem_stride
     */
    public static native void img_linear2D(long stream_address,
            float alpha, long dX_address, float beta,
            long dY_address,
            int lengthv, int mem_width, int mem_stride);

    /**
     * <pre>
     * (1) X1.width = X2.width = Y.width
     * (2) X1.[length, lengthv] = Y.[length, lengthv] = [length, lengthv]
     * (3) [length, lengthv] % X2.[length, lengthv] == 0
     * (4) reshape: Xrow2 -> Xrow2[Xrow2.length]
     * (5) reshape: X1, Y -> (X1, Y)[length/X2.length, X2.length]
     * (6) X1[i], Y[i] is the ith row vector of X1, Y:
     *      for i from 1 to X2_length:
     *          Y[i](float) = alpha*X1[i](byte) + beta*X2(float) + gamma.
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     *
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * [height, width] = [1024, 1024]: Time = 0.110000, Speed = 106.534081 GB/s
     * </pre>
     *
     * @param cudaStream_address
     * @param dX1_address
     * @param dX2_address
     * @param row_lengthv
     * @param alpha
     * @param dY_address
     * @param beta
     * @param lengthv
     * @param gamma
     * @param mem_width
     * @param mem_stride
     */
    public static native void img_linear_dual2D_row(long cudaStream_address,
            long dX1_address,
            long dX2_address, int row_lengthv,//row_lengthv = X2_lengthv
            float alpha, float beta, float gamma,
            long dY_address,
            int lengthv, int mem_width, int mem_stride);

    /**
     * <pre>
     * (1) X1.width = X2.width = Y.width
     * (2) X1.[length, lengthv] = Y.[length, lengthv] = [length, lengthv]
     * (3) [length, lengthv] % X2.[length, lengthv] == 0
     * (4) reshape: Xrow2 -> Xrow2[Xrow2.length]
     * (5) reshape: X1, Y -> (X1, Y)[length/X2.length, X2.length]
     * (6) X1[i], Y[i] is the ith row vector of X1, Y:
     *      for i from 1 to X2_length:
     *          Y[i](float) = alpha*X1[i](int8) + beta*X2(float) + gamma.
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     *
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * [height, width] = [1024, 1024]: Time = 0.110000, Speed = 106.534081 GB/s
     * </pre>
     *
     * @param cudaStream_address
     * @param dX1_address
     * @param dX2_address
     * @param row_lengthv
     * @param alpha
     * @param beta
     * @param gamma
     * @param dY_address
     * @param lengthv
     * @param width
     * @param stride
     */
    public static native void img_linear_dual2D_field(long cudaStream_address,
            long dX1_address,
            long dX2_address, int row_lengthv,
            float alpha, float beta, float gamma,
            long dY_address,
            int lengthv, int width, int stride);
    //</editor-fold>

    /**
     * <pre>
     * Threshold: Y = (a*x > v ?  v1 : v2).
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     * (4) the datatype of Y&X is int8
     *
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * [height, width] = [1024, 1024]: Time = 0.108000, Speed = 72.337959 GB/s
     * </pre>
     *
     * @param stream_address
     * @param alpha
     * @param dX_address
     * @param v
     * @param v1
     * @param v2
     * @param dY_address
     * @param lengthv
     * @param mem_width
     * @param mem_stride
     */
    public static native void img_threshold2D(long stream_address,
            long dX_address, float alpha, float v, byte v1, byte v2,
            long dY_address,
            int lengthv, int mem_width, int mem_stride);

    /**
     * <pre>
     * Linear Transformation:
     *  [1] Y = alpha * X + beta
     *  [2] Y = clip(Y, 0, 255).
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     * (4) the datatype of Y&X is int8
     *
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * [height, width] = [1024, 1024]: Time = 0.108000, Speed = 72.337959 GB/s
     * </pre>
     *
     * @param stream_address
     * @param alpha
     * @param dX_address
     * @param beta
     * @param gamma
     * @param dY_address
     * @param lengthv
     * @param mem_width
     * @param mem_stride
     */
    public static native void img_quadratic2D(long stream_address,
            long dX_address, float alpha, float beta, float gamma,
            long dY_address,
            int lengthv, int mem_width, int mem_stride);

    /**
     * <pre>
     *  [1] Y = C * log(alpha * X + beta)
     *  [2] Y = clip(Y, 0, 255).
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     * (4) the datatype of Y&X is int8
     * </pre>
     *
     * @param stream_address
     * @param C
     * @param alpha
     * @param dX_address
     * @param beta
     * @param dY_address
     * @param lengthv
     * @param mem_width
     * @param mem_stride
     */
    public static native void img_log2D(long stream_address,
            float C, float alpha, long dX_address, float beta,
            long dY_address,
            int lengthv, int mem_width, int mem_stride);

    /**
     * <pre>
     *  [1] Y = exp(alpha * X + beta) + C
     *  [2] Y = clip(Y, 0, 255).
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     * (4) the datatype of Y&X is int8
     * </pre>
     *
     * @param stream_address
     * @param alpha
     * @param dX_address
     * @param beta
     * @param C
     * @param dY_address
     * @param lengthv
     * @param mem_width
     * @param mem_stride
     */
    public static native void img_exp2D(long stream_address,
            float alpha, long dX_address, float beta, float C,
            long dY_address,
            int lengthv, int mem_width, int mem_stride);
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="image: pad, trim">
    /**
     * <pre>
     * X[N, IH, IW, IC] -> Y[N, OH, OW, OC]
     * (1) IH + ph0 + ph1 = OH
     * (2) IW + pw0 + pw1 = OW
     * (3) IC + pc0 + pc1 = OC.
     * (1) the datatype of Y&X is int8
     * (2) { IC， OC } % 4 == 0
     * </pre>
     *
     * @param stream_address
     * @param Y_address
     * @param OH
     * @param OW
     * @param OC
     * @param X_address
     * @param IH
     * @param IW
     * @param IC
     * @param N
     * @param ph0
     * @param pw0
     * @param pc0
     */
    public static native void img_pad(long stream_address,
            long Y_address, int OH, int OW, int OC,
            long X_address, int IH, int IW, int IC,
            int N, int ph0, int pw0, int pc0);

    /**
     * X[N, IH, IW, IC] -> Y[N, OH, OW, OC] (1) IH - ph0 - ph1 = OH (2) IW - pw0
     * - pw1 = OW (3) IC - pc0 - pc1 = OC. (1) the datatype of Y&X is int8 (2) {
     * IC， OC } % 4 == 0
     *
     * @param stream_address
     * @param Y_address
     * @param OH
     * @param OW
     * @param OC
     * @param X_address
     * @param IH
     * @param IW
     * @param IC
     * @param N
     * @param ph0
     * @param pw0
     * @param pc0
     */
    public static native void img_trim(long stream_address,
            long Y_address, int OH, int OW, int OC,
            long X_address, int IH, int IW, int IC,
            int N, int ph0, int pw0, int pc0);
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="image: transpose(2D -> 4D)">
    /**
     * <pre>
     * transpose2D: X[dim0, dim1] -> Y[dim1, dim0].
     * (1) mem_strideX = (Xdim1 + 3) / 4 * 4
     * (2) mem_strideY = (Ydim1 + 3) / 4 * 4
     * (3) Xdim1 = Ydim0, Xdim0 = Ydim1
     * (4) {X. Y}.datatype = uint8(pixel)
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
    public static native void img_transpose2D(long cudaStream_address,
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
     * (4) {X. Y}.datatype = uint8(pixel)
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
    public static native void img_transpose3D(long cudaStream_address,
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
     * (4) {X. Y}.datatype = uint8(pixel)
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
    public static native void img_transpose4D(long cudaStream_address,
            long dX_address, long dY_address,
            int Xdim1, int Xdim2, int Xdim3,
            int Ydim1, int Ydim2, int Ydim3,
            int dimIndex1, int dimIndex2,
            int mem_strideX, int mem_strideY,
            int length);
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="image: resize, affine">
    /**
     * <pre>
     * Change the size of Image:
     *  X[N, IH, IW, C] -> Y[N, OH, OW, C].
     * (1) the datatype of Y&X is int8
     * (2) C % 4 == 0
     * </pre>
     *
     * @param stream_address
     * @param dX_address
     * @param IH
     * @param IW
     * @param dY_address
     * @param OH
     * @param OW
     * @param N
     * @param C
     */
    public static native void img_resize(long stream_address,
            long dX_address, int IH, int IW,
            long dY_address, int OH, int OW,
            int N, int C);

    /**
     * <pre>
     * Affine Transformation of Image:
     *  [m00, m01, m02] ^-1    [r00, r01, r02]
     *  [m10, m11, m12]     => [r10, r11, r12]
     *  [  0,   0,   1]        [  0,   0,   1].
     * (1) the datatype of Y&X is int8
     * (2) C % 4 == 0
     *
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * [height, width] = [1024, 1024]: Time = 0.108000, Speed = 72.337959 GB/s
     * </pre>
     *
     * @param stream_address
     * @param dX_address
     * @param IH
     * @param IW
     * @param dY_address
     * @param OH
     * @param OW
     * @param r00
     * @param r01
     * @param r02
     * @param r10
     * @param r11
     * @param r12
     * @param N
     * @param C
     */
    public static native void img_affine(long stream_address,
            long dX_address, int IH, int IW,
            long dY_address, int OH, int OW,
            float r00, float r01, float r02,
            float r10, float r11, float r12,
            int N, int C);
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="image: gappedMemcpy, extract_3channels">
    /**
     * <pre>
     * |[width][gapX]|: width + gapX = strideX
     * |[width][gapY]|: width + gapY = strideY.
     * (1) (strideX, strideY) % 4 == 0
     * (2) (strideX, strideY) >= width
     * (3) the datatype of {X, Y} is unit8(pixel)
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
    public static native void img_gappedMemcpy2D(long stream_address,
            long dX_address, int Xstart, int strideX,
            long dY_address, int Ystart, int strideY,
            int width, int length);

    /**
     * <pre>
     * Extract 3 channels of X to construct Y:
     *  X(unit8)[N, H, W, C] -> Y(uint8)[N, H, W, 3].
     * (1) C % 4 == 0, X.stride = C
     * (2) Y.stride = 4
     * (3) lengthv = H * W * C, so: lengthv % 4 == 0
     * </pre>
     *
     * @param stream_address
     * @param dX_address
     * @param IC
     * @param dY_address
     * @param c0
     * @param c1
     * @param c2
     * @param lengthv
     */
    public static native void img_extract_3channels(long stream_address,
            long dX_address, int IC,
            long dY_address, int c0, int c1, int c2,
            int lengthv);
    //</editor-fold>
}
