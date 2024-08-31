/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.engine.cuda.impl.math;

import z.util.lang.annotation.Passed;

/**
 * <pre>
 * on GTX 1050, the greatest transfer speed of GPU is 112GB/s.
 * The biggest use radio: 112/88 = 0.7857. 
 * 
 * (1) regard a "tensor with its dimension greater than 1" as a matrix, we 
 * call the row vector of the matrix for short as "vector"
 * (2) obviously, the vector is a comprehensive concept for an element of
 * a tensor, in a 2D view.
 * </pre>
 * @author Gilgamesh
 */
public class Cuda_reduce 
{
    private Cuda_reduce() {}
    
    //<editor-fold defaultstate="collapsed" desc="common">
    public static int GRID_MAX = 8192;
    static final int MIN_int32(int a, int b) { return (a < b ? a : b); }
    
    public static final int straight_nextLengthV(int lengthv) {
	return (lengthv > 8192 ? MIN_int32((lengthv >> 13), GRID_MAX) : 1);
    }
    
    public static final int row_nextM(int M) { 
        return (M > 255 ? MIN_int32(M >> 8, GRID_MAX) : 1); 
    }
    
    public static final int center_nextN(int N, int M) {
	if (M > 15) {
            if (N > 63) return MIN_int32(N >> 6, GRID_MAX);
            if (N > 15) return 1;
	}
	if (M > 7) {
            if (N > 127) return MIN_int32(N >> 7, GRID_MAX);
            if (N >  31) return 1;
	}
	return MIN_int32((N + 63) >> 6, GRID_MAX);
    }
    
    public static final int field_nextN(int N, int M) {
	if (M > 15) {
            if (N > 63) return MIN_int32(N >> 6, GRID_MAX);
            if (N > 15) return 1;
	}
	if (M > 7) {
            if (N > 127) return MIN_int32(N >> 7, GRID_MAX);
            if (N >  31) return 1;
	}
	return MIN_int32((N + 63) >> 6, GRID_MAX);
    }
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="straight reduce">
    /**
     * <pre>
     * (1) width = (stride + 3)/4*4
     * (2) lengthv % stride = 0, lengthv = length / width * stride
     * (3) reshape: X -> Tensor1D[lengthv], lengthv = X.length
     * (4) binomialSum: result = sum(alpha*X + beta).
     *
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * for [height, width] from [100, 40] to(+1, +1) [105, 64]: correct
     * for [lengthv] = [1024*1024]: Time = 0.110000, Speed = 35.511360 GB/s
     * </pre>
     *
     * @param cudaStream_address
     * @param dX_address
     * @param alpha
     * @param beta
     * @param lengthv
     * @param dV_address
     * @param mem_width
     * @param mem_stride
     * @param partNum
     * @return
     */
    @Passed
    public static native int straight_linear(long cudaStream_address,
            long dX_address,
            float alpha, float beta,
            int lengthv,
            long dV_address,
            int mem_width, int mem_stride,
            int partNum);

    /**
     * <pre>
     * (1) width = (stride + 3)/4*4
     * (2) lengthv % stride = 0, lengthv = length / width * stride
     * (3) reshape: X -> Tensor1D[lengthv], lengthv = X.length
     * (4) binomialSum: result = sum(alpha*X^2[i] + beta*X[i] + gamma).
     *
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * for [height, width] from [100, 40] to(+1, +1) [105, 64]: correct
     * for [lengthv] = [1024*1024]: Time = 0.110000, Speed = 35.511360 GB/s
     * </pre>
     *
     * @param cudaStream_address
     * @param dX_address
     * @param alpha
     * @param beta
     * @param gamma
     * @param lengthv
     * @param dV_address
     * @param mem_width
     * @param mem_stride
     * @param partNum
     * @return
     */
    @Passed
    public static native int straight_quadratic(long cudaStream_address,
            long dX_address,
            float alpha, float beta, float gamma,
            int lengthv,
            long dV_address,
            int mem_width, int mem_stride,
            int partNum);

    /**
     * <pre>
     * (1) width = (stride + 3)/4*4
     * (2) lengthv % stride = 0, lengthv = length / width * stride
     * (3) reshape: X -> Tensor1D[lengthv], lengthv = X.length
     * (4) Maximum: result = max(X[i]).
     *
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * for [height, width] from [100, 40] to(+1, +1) [105, 64]: correct
     * for [lengthv] = [1024*1024]: Time = 0.094000, Speed = 41.555851 GB/s
     * </pre>
     *
     * @param cudaStream_address
     * @param dX_address
     * @param lengthv
     * @param dV_address
     * @param mem_width
     * @param mem_stride
     * @param partNum
     * @return
     */
    @Passed
    public static native int straight_max(long cudaStream_address,
            long dX_address, int lengthv,
            long dV_address,
            int mem_width, int mem_stride,
            int partNum);

    /**
     * <pre>
     * (1) width = (stride + 3)/4*4
     * (2) lengthv % stride = 0, lengthv = length / width * stride
     * (3) reshape: X -> Tensor1D[lengthv], lengthv = X.length
     * (4) Minimum: result = min(X[i]).
     *
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * for [height, width] from [100, 40] to(+1, +1) [105, 64]: correct
     * for [lengthv] = [1024*1024]: Time = 0.090000, Speed = 43.402775 GB/s
     * </pre>
     *
     * @param cudaStream_address
     * @param dX_address
     * @param lengthv
     * @param dV_address
     * @param mem_width
     * @param mem_stride
     * @param partNum
     * @return
     */
    @Passed
    public static native int straight_min(long cudaStream_address,
            long dX_address, int lengthv,
            long dV_address,
            int mem_width, int mem_stride,
            int partNum);

    /**
     * <pre>
     * (1) width = (stride + 3)/4*4
     * (2) lengthv % stride = 0, lengthv = length / width * stride
     * (3) reshape: X -> Tensor1D[lengthv], lengthv = X.length
     * (4) Maximum: result = max(X[i]).
     *
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * for [height, width] from [100, 40] to(+1, +1) [105, 64]: correct
     * for [lengthv] = [1024*1024]: Time = 0.094000, Speed = 41.555851 GB/s
     * </pre>
     *
     * @param cudaStream_address
     * @param dX_address
     * @param lengthv
     * @param dV_address
     * @param dIndex_address
     * @param mem_width
     * @param mem_stride
     * @param partNum
     * @return
     */
    @Passed
    public static native int straight_max_indexed(long cudaStream_address,
            long dX_address, int lengthv,
            long dV_address, long dIndex_address,
            int mem_width, int mem_stride,
            int partNum);

    /**
     * <pre>
     * (1) width = (stride + 3)/4*4
     * (2) lengthv % stride = 0, lengthv = length / width * stride
     * (3) reshape: X -> Tensor1D[lengthv], lengthv = X.length
     * (4) Minimum: result = min(X[i]).
     *
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * for [height, width] from [100, 40] to(+1, +1) [105, 64]: correct
     * for [lengthv] = [1024*1024]: Time = 0.090000, Speed = 43.402775 GB/s
     * </pre>
     *
     * @param cudaStream_address
     * @param dX_address
     * @param lengthv
     * @param dV_address
     * @param dIndex_address
     * @param mem_width
     * @param mem_stride
     * @param partNum
     * @return
     */
    @Passed
    public static native int straight_min_indexed(long cudaStream_address,
            long dX_address, int lengthv,
            long dV_address, long dIndex_address,
            int mem_width, int mem_stride,
            int partNum);
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="field reduce">
    //<editor-fold defaultstate="collapsed" desc="field linear">
    /**
     * <pre>
     * (1) stride = (width + 3)/4 * 4
     * (2) N = field_length
     * (3) M = row_lengthv(the lengthv of the reshaped row vector, which
     *  contains the result of this reduction)
     * (4) N*M = X.lengthv
     * (5) M % stride == 0
     * (6) reshape: X -> Tensor[N, M]
     *  find binomial summary of each field vector of X:
     *      for i from 1 to M:
     *          Y[i] = sum(alpha*X[i] + beta, 0, M).
     *
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * for [height, width] from [100, 40] to(+1, +1) [105, 64]: correct
     * [height, width] = [1024, 1024]: Time = 0.102000, Speed = 38.296566 GB/s
     * </pre>
     *
     * @param cudaStream_address
     * @param dX_address
     * @param alpha
     * @param beta
     * @param N
     * @param M
     * @param dV_address
     * @param dY_address
     * @param mem_width
     * @param mem_stride
     * @param partNum
     * @return
     */
    @Passed
    public static native int field_linear(long cudaStream_address,
            long dX_address,
            float alpha, float beta,
            int N, int M,
            long dV_address, long dY_address,
            int mem_width, int mem_stride,
            int partNum);

    /**
     * <pre>
     * (1) stride = (width + 3)/4 * 4
     * (2) N = field_length
     * (3) M = row_lengthv(the lengthv of the reshaped row vector, which
     *  contains the result of this reduction)
     * (4) N*M = X.lengthv
     * (5) M % stride == 0
     * (6) reshape: X -> Tensor[N, M]
     *  find binomial summary of each field vector of X:
     *      for i from 1 to M:
     *          Y[i] = sum(alpha*X1[i] + beta*X2[i] + gamma, 0, M).
     *
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * for [height, width] from [100, 40] to(+1, +1) [105, 64]: correct
     * [height, width] = [1024, 1024]: Time = 0.102000, Speed = 38.296566 GB/s
     * </pre>
     *
     * @param cudaStream_address
     * @param dX1_address
     * @param dX2_address
     * @param alpha
     * @param beta
     * @param gamma
     * @param N
     * @param M
     * @param dV_address
     * @param dY_address
     * @param mem_width
     * @param mem_stride
     * @param partNum
     * @return
     */
    @Passed
    public static native int field_linear_dual(long cudaStream_address,
            long dX1_address, long dX2_address,
            float alpha, float beta, float gamma,
            int N, int M,
            long dV_address, long dY_address,
            int mem_width, int mem_stride,
            int partNum);

    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="field quadratic">
    /**
     * <pre>
     * (1) stride = (width + 3)/4 * 4
     * (2) N = field_length
     * (3) M = row_lengthv(the lengthv of the reshaped row vector, which
     *  contains the result of this reduction)
     * (4) N*M = X.lengthv
     * (5) M % stride == 0
     * (6) reshape: X -> Tensor[N, M]
     *  find binomial summary of each field vector of X:
     *      for i from 1 to M:
     *          Y[i] = sum(alpha*X[i]^2 + beta*X[i] + gamma, 0, M).
     *
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * for [height, width] from [100, 40] to(+1, +1) [105, 64]: correct
     * [height, width] = [1024, 1024]: Time = 0.102000, Speed = 38.296566 GB/s
     * </pre>
     *
     * @param cudaStream_address
     * @param dX_address
     * @param alpha
     * @param beta
     * @param gamma
     * @param N
     * @param M
     * @param dV_address
     * @param dY_address
     * @param mem_width
     * @param mem_stride
     * @param partNum
     * @return
     */
    @Passed
    public static native int field_quadratic(long cudaStream_address,
            long dX_address,
            float alpha, float beta, float gamma,
            int N, int M,
            long dV_address, long dY_address,
            int mem_width, int mem_stride,
            int partNum);

    /**
     * <pre>
     * (1) stride = (width + 3)/4 * 4
     * (2) N = field_length
     * (3) M = row_lengthv(the lengthv of the reshaped row vector, which
     *  contains the result of this reduction)
     * (4) N*M = X.lengthv
     * (5) M % stride == 0
     * (6) reshape: X -> Tensor[N, M]
     *  find binomial summary of each field vector of X:
     *      for i from 1 to M:
     *          Y[i] = sum(k11*X1*X1 + k12*X1*X2 + k22*X2*X2 + K1*X1 + K2 *X2 + C, 0, M).
     *
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * for [height, width] from [100, 40] to(+1, +1) [105, 64]: correct
     * [height, width] = [1024, 1024]: Time = 0.102000, Speed = 38.296566 GB/s
     * </pre>
     *
     * @param cudaStream_address
     * @param dX1_address
     * @param dX2_address
     * @param N
     * @param k11
     * @param M
     * @param k12
     * @param dV_address
     * @param k22
     * @param dY_address
     * @param k1
     * @param k2
     * @param C
     * @param mem_width
     * @param mem_stride
     * @param partNum
     * @return
     */
    @Passed
    public static native int field_quadratic_dual(long cudaStream_address,
            long dX1_address, long dX2_address,
            float k11, float k12, float k22,
            float k1, float k2, float C,
            int N, int M,
            long dV_address, long dY_address,
            int mem_width, int mem_stride,
            int partNum);
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="field linear quadratic">
    /**
     * <pre>
     * (1) stride = (width + 3)/4 * 4
     * (2) N = field_length
     * (3) M = row_lengthv(the lengthv of the reshaped row vector, which
     *  contains the result of this reduction)
     * (4) N*M = X.lengthv
     * (5) M % stride == 0
     * (6) reshape: X -> Tensor[N, M]
     *  find binomial summary of each field vector of X:
     *      for i from 1 to M:
     *          Y1[i] = sum(alpha1*X[i] + beta1*X, 0, M)
     *          Y2[i] = sum(alpha2*X[i] + beta2*X[i] + gamma2, 0, M).
     *
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * for [height, width] from [100, 40] to(+1, +1) [105, 64]: correct
     * [height, width] = [1024, 1024]: Time = 0.102000, Speed = 38.296566 GB/s
     * </pre>
     *
     * @param stream1
     * @param stream2
     * @param dX_address
     * @param alpha1
     * @param beta1
     * @param alpha2
     * @param beta2
     * @param gamma2
     * @param N
     * @param dV1_address
     * @param M
     * @param dV2_address
     * @param dY2_address
     * @param dY1_address
     * @param mem_width
     * @param mem_stride
     * @param partNum
     * @return
     */
    @Passed
    public static native int field_linear_quadratic(long stream1, long stream2,
            long dX_address,
            float alpha1, float beta1,
            float alpha2, float beta2, float gamma2,
            int N, int M,
            long dV1_address, long dY1_address,
            long dV2_address, long dY2_address,
            int mem_width, int mem_stride,
            int partNum);

    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="field_linear2_square_row">
    /**
     * <pre>
     * (1) stride = (width + 3)/4 * 4
     * (2) N = field_length
     * (3) M = row_lengthv(the lengthv of the reshaped row vector, which
     *  contains the result of this reduction)
     * (4) N*M = X.lengthv
     * (5) M % stride == 0
     * (6) reshape: X -> Tensor[N, M]
     *  find binomial summary of each field vector of X:
     *      for i from 1 to M:
     *          Y[i] = sum(C * (alpha*X1[i] + beta*X2 + gamma)^2 + beta1*X, 0, M).
     *
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * for [height, width] from [100, 40] to(+1, +1) [105, 64]: correct
     * [height, width] = [1024, 1024]: Time = 0.102000, Speed = 38.296566 GB/s
     * </pre>
     *
     * @param stream
     * @param dX2_address
     * @param C
     * @param dX1_address
     * @param alpha
     * @param beta
     * @param gamma
     * @param N
     * @param dV_address
     * @param M
     * @param dY_address
     * @param mem_width
     * @param mem_stride
     * @param partNum
     * @return
     */
    @Passed
    public static native int field_linear2_square_row(long stream,
            long dX1_address, long dX2_address,
            float C, float alpha, float beta, float gamma,
            int N, int M,
            long dV_address, long dY_address,
            int mem_width, int mem_stride,
            int partNum);
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="field max, min">
    /**
     * <pre>
     * (1) stride = (width + 3)/4 * 4
     * (2) N = field_length
     * (3) M = row_lengthv(the lengthv of the reshaped row vector, which
     *  contains the result of this reduction)
     * (4) N*M = X.lengthv
     * (5) M % stride == 0
     * (6) reshape: X -> Tensor[N, M]
     *  find max factor of each field vector of X:
     *      for i from 1 to M:
     *          Y[i] = max(X[i], 0, M).
     *
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * for [height, width] from [100, 40] to(+1, +1) [105, 64]: correct
     * [height, width] = [1024, 1024]: Time = 0.102000, Speed = 38.296566 GB/s
     * </pre>
     *
     * @param cudaStream_address
     * @param dX_address
     * @param N
     * @param M
     * @param dV_address
     * @param dY_address
     * @param mem_width
     * @param mem_stride
     * @param partNum
     * @return
     */
    @Passed
    public static native int field_max(long cudaStream_address,
            long dX_address,
            int N, int M,
            long dV_address, long dY_address,
            int mem_width, int mem_stride,
            int partNum);

    /**
     * <pre>
     * (1) stride = (width + 3)/4 * 4
     * (2) N = field_length
     * (3) M = row_lengthv(the lengthv of the reshaped row vector, which
     *  contains the result of this reduction)
     * (4) N*M = X.lengthv
     * (5) M % stride == 0
     * (6) reshape: X -> Tensor[N, M]
     *  find m factor of each field vector of X:
     *      for i from 1 to M:
     *          Y[i] = min(X[i], 0, M).
     *
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * for [height, width] from [100, 40] to(+1, +1) [105, 64]: correct
     * [height, width] = [1024, 1024]: Time = 0.102000, Speed = 38.296566 GB/s
     * </pre>
     *
     * @param cudaStream_address
     * @param dX_address
     * @param N
     * @param M
     * @param dV_address
     * @param dY_address
     * @param mem_width
     * @param mem_stride
     * @param partNum
     * @return
     */
    @Passed
    public static native int field_min(long cudaStream_address,
            long dX_address,
            int N, int M,
            long dV_address, long dY_address,
            int mem_width, int mem_stride,
            int partNum);

    /**
     * <pre>
     * (1) stride = (width + 3)/4 * 4
     * (2) N = field_length
     * (3) M = row_lengthv(the lengthv of the reshaped row vector, which
     *  contains the result of this reduction)
     * (4) N*M = X.lengthv
     * (5) M % stride == 0
     * (6) reshape: X -> Tensor[N, M]
     *  find max factor of each field vector of X:
     *      for i from 1 to M:
     *          Y[i] = max(X[i], 0, M).
     *
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * for [height, width] from [100, 40] to(+1, +1) [105, 64]: correct
     * [height, width] = [1024, 1024]: Time = 0.102000, Speed = 38.296566 GB/s
     * </pre>
     *
     * @param cudaStream_address
     * @param dX_address
     * @param N
     * @param M
     * @param dV_address
     * @param dVIndex_address
     * @param dY_address
     * @param mem_width
     * @param dIndex_address
     * @param mem_stride
     * @param partNum
     * @return
     */
    @Passed
    public static native int field_max_indexed(long cudaStream_address,
            long dX_address,
            int N, int M,
            long dV_address, long dVIndex_address,
            long dY_address, long dIndex_address,
            int mem_width, int mem_stride,
            int partNum);

    /**
     * <pre>
     * (1) stride = (width + 3)/4 * 4
     * (2) N = field_length
     * (3) M = row_lengthv(the lengthv of the reshaped row vector, which
     *  contains the result of this reduction)
     * (4) N*M = X.lengthv
     * (5) M % stride == 0
     * (6) reshape: X -> Tensor[N, M]
     *  find m factor of each field vector of X:
     *      for i from 1 to M:
     *          Y[i] = min(X[i], 0, M).
     *
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * for [height, width] from [100, 40] to(+1, +1) [105, 64]: correct
     * [height, width] = [1024, 1024]: Time = 0.102000, Speed = 38.296566 GB/s
     * </pre>
     *
     * @param cudaStream_address
     * @param dX_address
     * @param N
     * @param M
     * @param dV_address
     * @param dVIndex_address
     * @param dY_address
     * @param mem_width
     * @param dIndex_address
     * @param mem_stride
     * @param partNum
     * @return
     */
    @Passed
    public static native int field_min_indexed(long cudaStream_address,
            long dX_address,
            int N, int M,
            long dV_address, long dVIndex_address,
            long dY_address, long dIndex_address,
            int mem_width, int mem_stride,
            int partNum);
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="field Affine reduce">
    //<editor-fold defaultstate="collapsed" desc="affine{deltaA, deltaB}">
    /**
     * <pre>
     * (1) stride = (width + 3)/4 * 4
     * (2) N = field_length
     * (3) M = row_lengthv(the lengthv of the reshaped row vector, which
     *  contains the result of this reduction)
     * (4) N*M = X.lengthv
     * (5) M % stride == 0
     * (6) BatchNorm:
     *      [1] X_var = X_square_mean - X_mean*X_mean
     *      [2] X_norm = (X - X_mean)/sqrt(X_var)
     *      [3] Y = X_norm * A + B -> X_norm = (Y - B)/A
     * (7) Filed the gradient of A of BatchNorm:
     *      [1] deltaA = sumOfEachField: deltaY * X_norm = deltaY * (Y - B)/A
     * (8) reshape: [X, Y] -> Tensor[N = field_length, M = row_lengthv]
     * (9) V1: holdY(), Y is not changed.
     *
     * ---Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * for [height, width] from [100, 40] to(+1, +1) [105, 64]: correct
     * [height, width] = [1024, 1024]:
     * </pre>
     *
     * @param cudaStream_address
     * @param d_deltaY_address
     * @param dY_address
     * @param dA_address
     * @param dB_address
     * @param N
     * @param M
     * @param d_deltaA_buf_address
     * @param d_deltaA_address
     * @param mem_width
     * @param mem_stride
     * @param partNum
     * @return
     */
    @Passed
    public static native int field_affine_deltaA_v1(long cudaStream_address,
            long d_deltaY_address,
            long dY_address,
            long dA_address, long dB_address,
            int N, int M,
            long d_deltaA_buf_address, long d_deltaA_address,//deltaA = deltaXp2
            int mem_width, int mem_stride,
            int partNum);

    /**
     * <pre>
     * (1) stride = (width + 3)/4 * 4
     * (2) N = field_length
     * (3) M = row_lengthv(the lengthv of the reshaped row vector, which
     *  contains the result of this reduction)
     * (4) N*M = X.lengthv
     * (5) M % stride == 0
     * (6) BatchNorm:
     *      [1] X_var = X_square_mean - X_mean*X_mean
     *      [2] X_norm = (X - X_mean)/sqrt(X_var)
     *      [3] Y = X_norm * A + B -> X_norm = (Y - B)/A
     * (7) Filed the gradient of A of BatchNorm:
     *      [1] deltaA = sumOfEachField: deltaY * X_norm = deltaY * (Y - B)/A
     *      [2] deltaB = sumOfEachField: deltaY
     * (8) reshape: [X, Y] -> Tensor[N = field_length, M = row_lengthv]
     * (9) V1: holdY(), Y is not changed.
     *
     * ---Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * for [height, width] from [100, 40] to(+1, +1) [105, 64]: correct
     * [height, width] = [1024, 1024]:
     * </pre>
     *
     * @param stream1
     * @param stream2
     * @param d_deltaY_address
     * @param dY_address
     * @param dA_address
     * @param dB_address
     * @param N
     * @param M
     * @param d_deltaA_buf_address
     * @param d_deltaA_address
     * @param d_deltaB_buf_address
     * @param d_deltaB_address
     * @param mem_width
     * @param mem_stride
     * @param partNum
     * @return
     */
    @Passed
    public static native int field_affine_deltaAB_v1(long stream1, long stream2,
            long d_deltaY_address,
            long dY_address,
            long dA_address, long dB_address,
            int N, int M,
            long d_deltaA_buf_address, long d_deltaA_address,//deltaA = deltaXp2
            long d_deltaB_buf_address, long d_deltaB_address,//deltaB = deltaXp1
            int mem_width, int mem_stride,
            int partNum);

    /**
     * <pre>
     * (1) stride = (width + 3)/4 * 4
     * (2) N = field_length
     * (3) M = row_lengthv(the lengthv of the reshaped row vector, which
     *  contains the result of this reduction)
     * (4) N*M = X.lengthv
     * (5) M % stride == 0
     * (6) BatchNorm:
     *      [1] X_var = X_square_mean - X_mean*X_mean
     *      [2] X_norm = (X - X_mean)/sqrt(X_var)
     *      [3] Y = X_norm * A + B -> X_norm = (Y - B)/A
     * (7) Filed the gradient of A of BatchNorm:
     *      [1] deltaA = sumOfEachField: deltaY * X_norm = deltaY * (Y - B)/A
     *      [2] deltaB = sumOfEachField: deltaY
     * (8) reshape: [X, Y] -> Tensor[N = field_length, M = row_lengthv]
     * (9) V2: holdX(), X is not changed.
     *
     * ---Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * for [height, width] from [100, 40] to(+1, +1) [105, 64]: correct
     * [height, width] = [1024, 1024]:
     * </pre>
     *
     * @param stream1
     * @param stream2
     * @param d_deltaY_address
     * @param dX_address
     * @param N
     * @param M
     * @param d_deltaA_buf_address
     * @param d_deltaA_address
     * @param d_deltaB_buf_address
     * @param d_deltaB_address
     * @param mem_width
     * @param mem_stride
     * @param partNum
     * @return
     */
    @Passed
    public static native int field_affine_deltaAB_v2(long stream1, long stream2,
            long d_deltaY_address,
            long dX_address,//Y = X_norm
            int N, int M,
            long d_deltaA_buf_address, long d_deltaA_address,//deltaA = deltaXp2
            long d_deltaB_buf_address, long d_deltaB_address,//deltaB = deltaXp1
            int mem_width, int mem_stride,
            int partNum);

    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="affine + leakyRelu{deltaA, deltaB}">
    /**
     * <pre>
     * (1) stride = (width + 3)/4 * 4
     * (2) N = field_length
     * (3) M = row_lengthv(the lengthv of the reshaped row vector, which
     *  contains the result of this reduction)
     * (4) N*M = X.lengthv
     * (5) M % stride == 0
     * (6) BatchNorm:
     *      [1] X_var = X_square_mean - X_mean*X_mean
     *      [2] X_norm = (X - X_mean)/sqrt(X_var)
     *      [3] Y1 = X_norm * A + B -> X_norm = (Y1 - B)/A
     *      [4] Y2 = leaky_relu(Y1)
     * (7) Filed the gradient of A of BatchNorm:
     *      [1] deltaY11 = leaky_relu(Y1)' * deltaY
     *      [2] deltaA = sumOfEachField: deltaY1 * X_norm = deltaY * (Y - B)/A
     *      [3] deltaB = sumOfEachField: deltaY1
     * (8) reshape: [X, Y] -> Tensor[N = field_length, M = row_lengthv]
     * (9) V1: holdY(), Y is not changed.
     *
     * ---Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * for [height, width] from [100, 40] to(+1, +1) [105, 64]: correct
     * [height, width] = [1024, 1024]:
     * </pre>
     *
     * @param stream1
     * @param stream2
     * @param d_deltaY_address
     * @param dY_address
     * @param k
     * @param dA_address
     * @param dB_address
     * @param N
     * @param M
     * @param d_deltaA_buf_address
     * @param d_deltaA_address
     * @param d_deltaB_buf_address
     * @param d_deltaB_address
     * @param mem_width
     * @param mem_stride
     * @param partNum
     * @return
     */
    public static native int field_affine_with_leakyRelu_deltaAB_v1(long stream1, long stream2,
            long d_deltaY_address,
            long dY_address, float k,
            long dA_address, long dB_address,
            int N, int M,
            long d_deltaA_buf_address, long d_deltaA_address,//deltaA = deltaXp2
            long d_deltaB_buf_address, long d_deltaB_address,//deltaB = deltaXp1
            int mem_width, int mem_stride,
            int partNum);

    /**
     * <pre>
     * (1) stride = (width + 3)/4 * 4
     * (2) N = field_length
     * (3) M = row_lengthv(the lengthv of the reshaped row vector, which
     *  contains the result of this reduction)
     * (4) N*M = X.lengthv
     * (5) M % stride == 0
     * (6) BatchNorm:
     *      [1] X_var = X_square_mean - X_mean*X_mean
     *      [2] X_norm = (X - X_mean)/sqrt(X_var)
     *      [3] Y1 = X_norm * A + B -> X_norm = (Y1 - B)/A
     *      [4] Y2 = leaky_relu(Y1)
     * (7) Filed the gradient of A of BatchNorm:
     *      [1] deltaY11 = leaky_relu(Y1)' * deltaY
     *      [2] deltaA = sumOfEachField: deltaY1 * X_norm = deltaY * (Y - B)/A
     *      [3] deltaB = sumOfEachField: deltaY1
     * (8) reshape: [X, Y] -> Tensor[N = field_length, M = row_lengthv]
     * (9) V1: holdY(), Y is not changed.
     *
     * ---Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * for [height, width] from [100, 40] to(+1, +1) [105, 64]: correct
     * [height, width] = [1024, 1024]:
     * </pre>
     *
     * @param stream1
     * @param stream2
     * @param d_deltaY_address
     * @param k
     * @param dX_address
     * @param dA_address
     * @param dB_address
     * @param N
     * @param M
     * @param d_deltaA_buf_address
     * @param d_deltaA_address
     * @param d_deltaB_buf_address
     * @param d_deltaB_address
     * @param mem_width
     * @param mem_stride
     * @param partNum
     * @return
     */
    public static native int field_affine_with_leakyRelu_deltaAB_v2(long stream1, long stream2,
            long d_deltaY_address, float k,
            long dX_address,
            long dA_address, long dB_address,
            int N, int M,
            long d_deltaA_buf_address, long d_deltaA_address,//deltaA = deltaXp2
            long d_deltaB_buf_address, long d_deltaB_address,//deltaB = deltaXp1
            int mem_width, int mem_stride,
            int partNum);
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="affine + function{deltaA, deltaB}">
    /**
     * <pre>
     * (1) stride = (width + 3)/4 * 4
     * (2) N = field_length
     * (3) M = row_lengthv(the lengthv of the reshaped row vector, which
     *  contains the result of this reduction)
     * (4) N*M = X.lengthv
     * (5) M % stride == 0
     * (6) BatchNorm:
     *      [1] X_var = X_square_mean - X_mean*X_mean
     *      [2] X_norm = (X - X_mean)/sqrt(X_var)
     *      [3] Y1 = X_norm * A + B -> X_norm = (Y1 - B)/A
     *      [4] Y2 = leaky_relu(Y1)
     * (7) Filed the gradient of A of BatchNorm:
     *      [1] deltaY11 = leaky_relu(Y1)' * deltaY
     *      [2] deltaA = sumOfEachField: deltaY1 * X_norm = deltaY * (Y - B)/A
     *      [3] deltaB = sumOfEachField: deltaY1
     * (8) reshape: [X, Y] -> Tensor[N = field_length, M = row_lengthv]
     * (9) V1: holdY(), Y is not changed.
     *
     * ---Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * for [height, width] from [100, 40] to(+1, +1) [105, 64]: correct
     * [height, width] = [1024, 1024]:
     * </pre>
     *
     * @param stream1
     * @param stream2
     * @param d_deltaY_address
     * @param dY_address
     * @param dA_address
     * @param dB_address
     * @param N
     * @param M
     * @param d_deltaA_buf_address
     * @param d_deltaA_address
     * @param d_deltaB_buf_address
     * @param d_deltaB_address
     * @param mem_width
     * @param mem_stride
     * @param partNum
     * @param func_type
     * @param func_params
     * @param func_params_length
     * @return
     */
    public static native int field_affine_with_function_deltaAB_v1(long stream1, long stream2,
            long d_deltaY_address,
            long dY_address,
            long dA_address, long dB_address,
            int N, int M,
            long d_deltaA_buf_address, long d_deltaA_address,//deltaA = deltaXp2
            long d_deltaB_buf_address, long d_deltaB_address,//deltaB = deltaXp1
            int mem_width, int mem_stride,
            int partNum,
            int func_type, float[] func_params, int func_params_length);

    /**
     * <pre>
     * (1) stride = (width + 3)/4 * 4
     * (2) N = field_length
     * (3) M = row_lengthv(the lengthv of the reshaped row vector, which
     *  contains the result of this reduction)
     * (4) N*M = X.lengthv
     * (5) M % stride == 0
     * (6) BatchNorm:
     *      [1] X_var = X_square_mean - X_mean*X_mean
     *      [2] X_norm = (X - X_mean)/sqrt(X_var)
     *      [3] Y1 = X_norm * A + B -> X_norm = (Y1 - B)/A
     *      [4] Y2 = leaky_relu(Y1)
     * (7) Filed the gradient of A of BatchNorm:
     *      [1] deltaY11 = leaky_relu(Y1)' * deltaY
     *      [2] deltaA = sumOfEachField: deltaY1 * X_norm = deltaY * (Y - B)/A
     *      [3] deltaB = sumOfEachField: deltaY1
     * (8) reshape: [X, Y] -> Tensor[N = field_length, M = row_lengthv]
     * (9) V1: holdY(), Y is not changed.
     *
     * ---Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * for [height, width] from [100, 40] to(+1, +1) [105, 64]: correct
     * [height, width] = [1024, 1024]:
     * </pre>
     *
     * @param stream1
     * @param stream2
     * @param d_deltaY_address
     * @param dX_address
     * @param dA_address
     * @param dB_address
     * @param N
     * @param M
     * @param d_deltaA_buf_address
     * @param d_deltaA_address
     * @param d_deltaB_buf_address
     * @param d_deltaB_address
     * @param mem_width
     * @param mem_stride
     * @param partNum
     * @param func_type
     * @param func_params
     * @param func_params_length
     * @return
     */
    public static native int field_affine_with_function_deltaAB_v2(long stream1, long stream2,
            long d_deltaY_address,
            long dX_address,
            long dA_address, long dB_address,
            int N, int M,
            long d_deltaA_buf_address, long d_deltaA_address,//deltaA = deltaXp2
            long d_deltaB_buf_address, long d_deltaB_address,//deltaB = deltaXp1
            int mem_width, int mem_stride,
            int partNum,
            int func_type, float[] func_params, int func_params_length);
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="sqBatchNorm{deltaA, deltaB}">
    /**
     * <pre>
     * (1) stride = (width + 3)/4 * 4
     * (2) N = field_length
     * (3) M = row_lengthv(the lengthv of the reshaped row vector, which
     *  contains the result of this reduction)
     * (4) N*M = X.lengthv
     * (5) M % stride == 0
     * (6) BatchNorm:
     *      [1] X_var = X_sqmean - X_mean*X_mean
     *      [2] X_norm = (X - X_mean) / [sqrt(X_var) + eps]
     * (7) Filed the gradient of A of BatchNorm:
     *      [1] deltaA = sumOfEachField:
     *          deltaY * X_norm = deltaY * (X - X_mean) / [sqrt(X_var) + eps]
     * (8) reshape: [X, Y] -> Tensor[N = field_length, M = row_lengthv].
     *
     * ---Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * for [height, width] from [100, 40] to(+1, +1) [105, 64]: correct
     * [height, width] = [1024, 1024]:
     * </pre>
     *
     * @param cudaStream_address
     * @param d_deltaY_address
     * @param dX_address
     * @param dX_mean_address
     * @param dX_sqmean_address
     * @param eps
     * @param N
     * @param M
     * @param d_deltaA_buf_address
     * @param d_deltaA_address
     * @param mem_width
     * @param mem_stride
     * @param partNum
     * @return
     */
    public static native int field_sqBatchNorm_deltaA_v2(long cudaStream_address,
            long d_deltaY_address,
            long dX_address,
            long dX_mean_address, long dX_sqmean_address, float eps,
            int N, int M,
            long d_deltaA_buf_address, long d_deltaA_address,//deltaA = deltaXp2
            int mem_width, int mem_stride,
            int partNum);

    /**
     * <pre>
     * (1) stride = (width + 3)/4 * 4
     * (2) N = field_length
     * (3) M = row_lengthv(the lengthv of the reshaped row vector, which
     *  contains the result of this reduction)
     * (4) N*M = X.lengthv
     * (5) M % stride == 0
     * (6) BatchNorm:
     *      [1] X_var = X_sqmean - X_mean*X_mean
     *      [2] X_norm = (X - X_mean) / [sqrt(X_var) + eps]
     * (7) Filed the gradient of A of BatchNorm:
     *      [1] deltaA = sumOfEachField:
     *              deltaY * X_norm = deltaY * (X - X_mean) / [sqrt(X_var) + eps]
     *      [2] deltaB = sumOfEachField: deltaY
     * (8) reshape: [X, Y] -> Tensor[N = field_length, M = row_lengthv].
     *
     *  ---Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * for [height, width] from [100, 40] to(+1, +1) [105, 64]: correct
     * [height, width] = [1024, 1024]:
     * </pre>
     *
     * @param stream1
     * @param stream2
     * @param d_deltaY_address
     * @param dX_address
     * @param dX_mean_address
     * @param dX_sqmean_address
     * @param eps
     * @param N
     * @param M
     * @param d_deltaA_buf_address
     * @param d_deltaA_address
     * @param d_deltaB_buf_address
     * @param d_deltaB_address
     * @param mem_width
     * @param mem_stride
     * @param partNum
     * @return
     */
    public static native int field_sqBatchNorm_deltaAB_v2(long stream1, long stream2,
            long d_deltaY_address,
            long dX_address,
            long dX_mean_address, long dX_sqmean_address, float eps,
            int N, int M,
            long d_deltaA_buf_address, long d_deltaA_address,//deltaA = deltaXp2
            long d_deltaB_buf_address, long d_deltaB_address,//deltaB = deltaXp1
            int mem_width, int mem_stride,
            int partNum);
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="batchNorm{deltaA, deltaB}">
    /**
     * <pre>
     * (1) stride = (width + 3)/4 * 4
     * (2) N = field_length
     * (3) M = row_lengthv(the lengthv of the reshaped row vector, which
     *  contains the result of this reduction)
     * (4) N*M = X.lengthv
     * (5) M % stride == 0
     * (6) BatchNorm:
     *      [1] X_var = X_sqmean - X_mean*X_mean
     *      [2] X_norm = (X - X_mean) / [sqrt(X_var) + eps]
     * (7) Filed the gradient of A of BatchNorm:
     *      [1] deltaA = sumOfEachField:
     *          deltaY * X_norm = deltaY * (X - X_mean) / [sqrt(X_var) + eps]
     * (8) reshape: [X, Y] -> Tensor[N = field_length, M = row_lengthv].
     *
     *  ---Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * for [height, width] from [100, 40] to(+1, +1) [105, 64]: correct
     * [height, width] = [1024, 1024]:
     * </pre>
     *
     * @param cudaStream_address
     * @param d_deltaY_address
     * @param dX_address
     * @param dX_mean_address
     * @param dX_var_address
     * @param eps
     * @param N
     * @param M
     * @param d_deltaA_buf_address
     * @param d_deltaA_address
     * @param mem_width
     * @param mem_stride
     * @param partNum
     * @return
     */
    public static native int field_batchNorm_deltaA_v2(long cudaStream_address,
            long d_deltaY_address,
            long dX_address,
            long dX_mean_address, long dX_var_address, float eps,
            int N, int M,
            long d_deltaA_buf_address, long d_deltaA_address,//deltaA = deltaXp2
            int mem_width, int mem_stride,
            int partNum);

    /**
     * <pre>
     * (1) stride = (width + 3)/4 * 4
     * (2) N = field_length
     * (3) M = row_lengthv(the lengthv of the reshaped row vector, which
     *  contains the result of this reduction)
     * (4) N*M = X.lengthv
     * (5) M % stride == 0
     * (6) BatchNorm:
     *      [1] X_var = X_sqmean - X_mean*X_mean
     *      [2] X_norm = (X - X_mean) / [sqrt(X_var) + eps]
     * (7) Filed the gradient of A of BatchNorm:
     *      [1] deltaA = sumOfEachField:
     *              deltaY * X_norm = deltaY * (X - X_mean) / [sqrt(X_var) + eps]
     *      [2] deltaB = sumOfEachField: deltaY
     * (8) reshape: [X, Y] -> Tensor[N = field_length, M = row_lengthv].
     *
     *  ---Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * for [height, width] from [100, 40] to(+1, +1) [105, 64]: correct
     * [height, width] = [1024, 1024]:
     * </pre>
     *
     * @param stream1
     * @param stream2
     * @param d_deltaY_address
     * @param dX_address
     * @param dX_mean_address
     * @param dX_var_address
     * @param eps
     * @param N
     * @param M
     * @param d_deltaA_buf_address
     * @param d_deltaA_address
     * @param d_deltaB_buf_address
     * @param d_deltaB_address
     * @param mem_width
     * @param mem_stride
     * @param partNum
     * @return
     */
    public static native int field_batchNorm_deltaAB_v2(long stream1, long stream2,
            long d_deltaY_address,
            long dX_address,
            long dX_mean_address, long dX_var_address, float eps,
            int N, int M,
            long d_deltaA_buf_address, long d_deltaA_address,//deltaA = deltaXp2
            long d_deltaB_buf_address, long d_deltaB_address,//deltaB = deltaXp1
            int mem_width, int mem_stride,
            int partNum);

    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="batchNorm + leakyRelu{deltaXp1, deltaXp2}">
    /**
     * <pre>
     * (1) stride = (width + 3)/4 * 4
     * (2) N = field_length
     * (3) M = row_lengthv(the lengthv of the reshaped row vector, which
     *  contains the result of this reduction)
     * (4) N*M = X.lengthv
     * (5) M % stride == 0
     * (6) BatchNorm:
     *      [1] X_var = X_sqmean - X_mean*X_mean
     *      [2] X_norm = (X - X_mean) / [sqrt(X_var) + eps]
     * (7) Filed the gradient of A of BatchNorm:
     *      [1] deltaA = sumOfEachField:
     *              deltaY * X_norm = deltaY * (X - X_mean) / [sqrt(X_var) + eps]
     *      [2] deltaB = sumOfEachField: deltaY
     * (8) reshape: [X, Y] -> Tensor[N = field_length, M = row_lengthv].
     *
     *  ---Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * for [height, width] from [100, 40] to(+1, +1) [105, 64]: correct
     * [height, width] = [1024, 1024]:
     * </pre>
     *
     * @param stream1
     * @param stream2
     * @param d_deltaY_address
     * @param dY_address
     * @param k
     * @param N
     * @param M
     * @param d_deltaXp1_buf_address
     * @param d_deltaXp1_address
     * @param d_deltaXp2_buf_address
     * @param d_deltaXp2_address
     * @param mem_width
     * @param mem_stride
     * @param partNum
     * @return
     */
    public static native int field_batchNorm_with_leakyRelu_deltaXp_v1(long stream1, long stream2,
            long d_deltaY_address,
            long dY_address, float k,
            int N, int M,
            long d_deltaXp1_buf_address, long d_deltaXp1_address,
            long d_deltaXp2_buf_address, long d_deltaXp2_address,
            int mem_width, int mem_stride,
            int partNum);

    /**
     * <pre>
     * (1) stride = (width + 3)/4 * 4
     * (2) N = field_length
     * (3) M = row_lengthv(the lengthv of the reshaped row vector, which
     *  contains the result of this reduction)
     * (4) N*M = X.lengthv
     * (5) M % stride == 0
     * (6) BatchNorm:
     *      [1] X_var = X_sqmean - X_mean*X_mean
     *      [2] X_norm = (X - X_mean) / [sqrt(X_var) + eps]
     * (7) Filed the gradient of A of BatchNorm:
     *      [1] deltaA = sumOfEachField:
     *              deltaY * X_norm = deltaY * (X - X_mean) / [sqrt(X_var) + eps]
     *      [2] deltaB = sumOfEachField: deltaY
     * (8) reshape: [X, Y] -> Tensor[N = field_length, M = row_lengthv].
     *
     *  ---Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * for [height, width] from [100, 40] to(+1, +1) [105, 64]: correct
     * [height, width] = [1024, 1024]:
     * </pre>
     *
     * @param stream1
     * @param stream2
     * @param d_deltaY_address
     * @param k
     * @param dX_address
     * @param dX_mean_address
     * @param dX_var_address
     * @param eps
     * @param N
     * @param M
     * @param d_deltaXp1_buf_address
     * @param d_deltaXp1_address
     * @param d_deltaXp2_buf_address
     * @param d_deltaXp2_address
     * @param mem_width
     * @param mem_stride
     * @param partNum
     * @return
     */
    public static native int field_batchNorm_with_leakyRelu_deltaXp_v2(long stream1, long stream2,
            long d_deltaY_address, float k,
            long dX_address,
            long dX_mean_address, long dX_var_address, float eps,
            int N, int M,
            long d_deltaXp1_buf_address, long d_deltaXp1_address,
            long d_deltaXp2_buf_address, long d_deltaXp2_address,
            int mem_width, int mem_stride,
            int partNum);
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="batchNorm + leakyRelu{deltaA, deltaB}">
    /**
     * <pre>
     * (1) stride = (width + 3)/4 * 4
     * (2) N = field_length
     * (3) M = row_lengthv(the lengthv of the reshaped row vector, which
     *  contains the result of this reduction)
     * (4) N*M = X.lengthv
     * (5) M % stride == 0
     * (6) BatchNorm:
     *      [1] X_var = X_sqmean - X_mean*X_mean
     *      [2] X_norm = (X - X_mean) / [sqrt(X_var) + eps]
     * (7) Filed the gradient of A of BatchNorm:
     *      [1] deltaA = sumOfEachField:
     *              deltaY * X_norm = deltaY * (X - X_mean) / [sqrt(X_var) + eps]
     *      [2] deltaB = sumOfEachField: deltaY
     * (8) reshape: [X, Y] -> Tensor[N = field_length, M = row_lengthv].
     *
     *  ---Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * for [height, width] from [100, 40] to(+1, +1) [105, 64]: correct
     * [height, width] = [1024, 1024]:
     * </pre>
     *
     * @param stream1
     * @param stream2
     * @param d_deltaY_address
     * @param k
     * @param dX_address
     * @param dX_mean_address
     * @param dX_var_address
     * @param eps
     * @param dA_address
     * @param dB_address
     * @param N
     * @param M
     * @param d_deltaA_buf_address
     * @param d_deltaA_address
     * @param d_deltaB_buf_address
     * @param d_deltaB_address
     * @param mem_width
     * @param mem_stride
     * @param partNum
     * @return
     */
    public static native int field_batchNorm_with_leakyRelu_deltaAB_v2(long stream1, long stream2,
            long d_deltaY_address, float k,
            long dX_address,
            long dX_mean_address, long dX_var_address, float eps,
            long dA_address, long dB_address,
            int N, int M,
            long d_deltaA_buf_address, long d_deltaA_address,//deltaA = deltaXp2
            long d_deltaB_buf_address, long d_deltaB_address,//deltaB = deltaXp1
            int mem_width, int mem_stride,
            int partNum);
    //</editor-fold>   
    //<editor-fold defaultstate="collapsed" desc="batchNorm + function{deltaXp1, deltaXp2}">
    /**
     * <pre>
     * (1) stride = (width + 3)/4 * 4
     * (2) N = field_length
     * (3) M = row_lengthv(the lengthv of the reshaped row vector, which
     *  contains the result of this reduction)
     * (4) N*M = X.lengthv
     * (5) M % stride == 0
     * (6) BatchNorm:
     *      [1] X_var = X_sqmean - X_mean*X_mean
     *      [2] X_norm = (X - X_mean) / [sqrt(X_var) + eps]
     * (7) Filed the gradient of A of BatchNorm:
     *      [1] deltaA = sumOfEachField:
     *              deltaY * X_norm = deltaY * (X - X_mean) / [sqrt(X_var) + eps]
     *      [2] deltaB = sumOfEachField: deltaY
     * (8) reshape: [X, Y] -> Tensor[N = field_length, M = row_lengthv].
     *
     *  ---Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * for [height, width] from [100, 40] to(+1, +1) [105, 64]: correct
     * [height, width] = [1024, 1024]:
     * </pre>
     *
     * @param stream1
     * @param stream2
     * @param d_deltaY_address
     * @param dY_address
     * @param N
     * @param M
     * @param d_deltaXp1_buf_address
     * @param d_deltaXp1_address
     * @param d_deltaXp2_buf_address
     * @param d_deltaXp2_address
     * @param mem_width
     * @param mem_stride
     * @param partNum
     * @param func_type
     * @param func_params
     * @param func_params_length
     * @return
     */
    public static native int field_batchNorm_with_function_deltaXp_v1(long stream1, long stream2,
            long d_deltaY_address,
            long dY_address,
            int N, int M,
            long d_deltaXp1_buf_address, long d_deltaXp1_address,
            long d_deltaXp2_buf_address, long d_deltaXp2_address,
            int mem_width, int mem_stride,
            int partNum,
            int func_type, float[] func_params, int func_params_length);

    /**
     * <pre>
     * (1) stride = (width + 3)/4 * 4
     * (2) N = field_length
     * (3) M = row_lengthv(the lengthv of the reshaped row vector, which
     *  contains the result of this reduction)
     * (4) N*M = X.lengthv
     * (5) M % stride == 0
     * (6) BatchNorm:
     *      [1] X_var = X_sqmean - X_mean*X_mean
     *      [2] X_norm = (X - X_mean) / [sqrt(X_var) + eps]
     * (7) Filed the gradient of A of BatchNorm:
     *      [1] deltaA = sumOfEachField:
     *              deltaY * X_norm = deltaY * (X - X_mean) / [sqrt(X_var) + eps]
     *      [2] deltaB = sumOfEachField: deltaY
     * (8) reshape: [X, Y] -> Tensor[N = field_length, M = row_lengthv].
     *
     *  ---Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * for [height, width] from [100, 40] to(+1, +1) [105, 64]: correct
     * [height, width] = [1024, 1024]:
     * </pre>
     *
     * @param stream1
     * @param stream2
     * @param d_deltaY_address
     * @param dX_address
     * @param dX_mean_address
     * @param dX_var_address
     * @param eps
     * @param N
     * @param M
     * @param d_deltaXp1_buf_address
     * @param d_deltaXp1_address
     * @param d_deltaXp2_buf_address
     * @param d_deltaXp2_address
     * @param mem_width
     * @param mem_stride
     * @param partNum
     * @param func_type
     * @param func_params
     * @param func_params_length
     * @return
     */
    public static native int field_batchNorm_with_function_deltaXp_v2(long stream1, long stream2,
            long d_deltaY_address,
            long dX_address,
            long dX_mean_address, long dX_var_address, float eps,
            int N, int M,
            long d_deltaXp1_buf_address, long d_deltaXp1_address,
            long d_deltaXp2_buf_address, long d_deltaXp2_address,
            int mem_width, int mem_stride,
            int partNum,
            int func_type, float[] func_params, int func_params_length);
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="batchNorm + function{deltaA, deltaB}">
    /**
     * <pre>
     * (1) stride = (width + 3)/4 * 4
     * (2) N = field_length
     * (3) M = row_lengthv(the lengthv of the reshaped row vector, which
     *  contains the result of this reduction)
     * (4) N*M = X.lengthv
     * (5) M % stride == 0
     * (6) BatchNorm:
     *      [1] X_var = X_sqmean - X_mean*X_mean
     *      [2] X_norm = (X - X_mean) / [sqrt(X_var) + eps]
     * (7) Filed the gradient of A of BatchNorm:
     *      [1] deltaA = sumOfEachField:
     *              deltaY * X_norm = deltaY * (X - X_mean) / [sqrt(X_var) + eps]
     *      [2] deltaB = sumOfEachField: deltaY
     * (8) reshape: [X, Y] -> Tensor[N = field_length, M = row_lengthv].
     *
     *  ---Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * for [height, width] from [100, 40] to(+1, +1) [105, 64]: correct
     * [height, width] = [1024, 1024]:
     * </pre>
     *
     * @param stream1
     * @param stream2
     * @param d_deltaY_address
     * @param dX_address
     * @param dX_mean_address
     * @param dX_var_address
     * @param eps
     * @param dA_address
     * @param dB_address
     * @param N
     * @param M
     * @param d_deltaA_buf_address
     * @param d_deltaA_address
     * @param d_deltaB_buf_address
     * @param d_deltaB_address
     * @param mem_width
     * @param mem_stride
     * @param partNum
     * @param func_type
     * @param func_params
     * @param func_params_length
     * @return
     */
    public static native int field_batchNorm_with_function_deltaAB_v2(long stream1, long stream2,
            long d_deltaY_address,
            long dX_address,
            long dX_mean_address, long dX_var_address, float eps,
            long dA_address, long dB_address,
            int N, int M,
            long d_deltaA_buf_address, long d_deltaA_address,//deltaA = deltaXp2
            long d_deltaB_buf_address, long d_deltaB_address,//deltaB = deltaXp1
            int mem_width, int mem_stride,
            int partNum,
            int func_type, float[] func_params, int func_params_length);
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="layerNorm{deltaA, deltaB}">
    /**
     * <pre>
     * (1) stride = (width + 3)/4 * 4
     * (2) N = field_length
     * (3) M = row_lengthv(the lengthv of the reshaped row vector, which
     *  contains the result of this reduction)
     * (4) N*M = X.lengthv
     * (5) M % stride == 0
     * (6) LayerNorm:
     *      [1] X_var = X_square_mean - X_mean*X_mean
     *      [2] X_norm = (X - X_mean) / [sqrt(X_var) + eps]
     * (7) Filed the gradient of A of BatchNorm:
     *      [1] deltaA = sumOfEachField:
     *          deltaY * X_norm = deltaY * (X - X_mean) / [sqrt(X_var) + eps]
     * (8) reshape: [X, Y] -> Tensor[N = field_length, M = row_lengthv].
     *
     *  ---Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * for [height, width] from [100, 40] to(+1, +1) [105, 64]: correct
     * [height, width] = [1024, 1024]:
     * </pre>
     *
     * @param cudaStream_address
     * @param d_deltaY_address
     * @param dX_address
     * @param dX_mean_address
     * @param dX_squareMean_address
     * @param eps
     * @param N
     * @param M
     * @param d_deltaA_buf_address
     * @param d_deltaA_address
     * @param mem_width
     * @param mem_stride
     * @param partNum
     * @return
     */
    public static native int field_layerNorm_deltaA_v2(long cudaStream_address,
            long d_deltaY_address,
            long dX_address,
            long dX_mean_address,
            long dX_squareMean_address, float eps,
            int N, int M,
            long d_deltaA_buf_address, long d_deltaA_address,
            int mem_width, int mem_stride,
            int partNum);

    /**
     * <pre>
     * (1) stride = (width + 3)/4 * 4
     * (2) N = field_length
     * (3) M = row_lengthv(the lengthv of the reshaped row vector, which
     *  contains the result of this reduction)
     * (4) N*M = X.lengthv
     * (5) M % stride == 0
     * (6) LayerNorm:
     *      [1] X_var = X_square_mean - X_mean*X_mean
     *      [2] X_norm = (X - X_mean) / [sqrt(X_var) + eps]
     * (7) Filed the gradient of A of BatchNorm:
     *      [1] deltaA = sumOfEachField:
     *              deltaY * X_norm = deltaY * (X - X_mean) / [sqrt(X_var) + eps]
     *      [2] deltaB = sumOfEachField: deltaY
     * (8) reshape: [X, Y] -> Tensor[N = field_length, M = row_lengthv].
     *
     *  ---Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * for [height, width] from [100, 40] to(+1, +1) [105, 64]: correct
     * [height, width] = [1024, 1024]:
     * </pre>
     *
     * @param stream1
     * @param stream2
     * @param d_deltaY_address
     * @param dX_address
     * @param dX_mean_address
     * @param dX_squareMean_address
     * @param eps
     * @param N
     * @param M
     * @param d_deltaA_buf_address
     * @param d_deltaA_address
     * @param d_deltaB_buf_address
     * @param d_deltaB_address
     * @param mem_width
     * @param mem_stride
     * @param partNum
     * @return
     */
    public static native int field_layerNorm_deltaAB_v2(long stream1, long stream2,
            long d_deltaY_address,
            long dX_address,
            long dX_mean_address,
            long dX_squareMean_address, float eps,
            int N, int M,
            long d_deltaA_buf_address, long d_deltaA_address,
            long d_deltaB_buf_address, long d_deltaB_address,
            int mem_width, int mem_stride,
            int partNum);
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="center reduce">
    /**
     * <pre>
     * (1) stride = (width + 3)/4 * 4
     * (2) X[dim0, dim1, dim2] -> Y[dim0, part, dim2]
     * (3) accumulate along dim1 axis:
     *      Y[i] = sum(alpha*X + beta).
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * for [height, width] from [100, 40] to(+1, +1) [105, 64]: correct
     * [height, width] = [1024, 1024]: Time = 0.102000, Speed = 38.296566 GB/s
     * </pre>
     *
     * @param stream_address
     * @param dX_address
     * @param alpha
     * @param beta
     * @param dim0
     * @param dim1
     * @param dim2
     * @param dV_address
     * @param dY_address
     * @param mem_width
     * @param mem_stride
     * @param partNum
     * @return
     */
    @Passed
    public static native int center_linear(long stream_address,
            long dX_address,
            float alpha, float beta,
            int dim0, int dim1, int dim2,
            long dV_address, long dY_address,
            int mem_width, int mem_stride,
            int partNum);
    
    /**
     * <pre>
     * (1) stride = (width + 3)/4 * 4
     * (2) X[dim0, dim1, dim2] -> Y[dim0, part, dim2]
     * (3) accumulate along dim1 axis:
     *      Y[i] = sum(alpha*X^2 + beta*X + gamma).
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * for [height, width] from [100, 40] to(+1, +1) [105, 64]: correct
     * [height, width] = [1024, 1024]: Time = 0.102000, Speed = 38.296566 GB/s
     * </pre>
     *
     * @param stream_address
     * @param dX_address
     * @param alpha
     * @param beta
     * @param gamma
     * @param dim0
     * @param dim1
     * @param dim2
     * @param dV_address
     * @param dY_address
     * @param mem_width
     * @param mem_stride
     * @param partNum
     * @return
     */
    @Passed
    public static native int center_quadratic(long stream_address,
            long dX_address,
            float alpha, float beta, float gamma,
            int dim0, int dim1, int dim2,
            long dV_address, long dY_address,
            int mem_width, int mem_stride,
            int partNum);
    
    //<editor-fold defaultstate="collapsed" desc="center_quadratic_dual">
    /**
     * <pre>
     * (1) stride = (width + 3)/4 * 4
     * (2) X[dim0, dim1, dim2] -> Y[dim0, part, dim2]
     * (3) accumulate along dim1 axis:
     *      Y[i] = sum(k11*X1*X1 + k12*X1*X2 + k22*X2*X2 + K1*X1 + K2 *X2 + C, 0, M).
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * for [height, width] from [100, 40] to(+1, +1) [105, 64]: correct
     * [height, width] = [1024, 1024]: Time = 0.102000, Speed = 38.296566 GB/s
     * </pre>
     *
     * @param stream_address
     * @param dX1_address
     * @param dX2_address
     * @param k11
     * @param k12
     * @param k22
     * @param k1
     * @param k2
     * @param C
     * @param dim0
     * @param dim1
     * @param dim2
     * @param dV_address
     * @param dY_address
     * @param mem_width
     * @param mem_stride
     * @param partNum
     * @return
     */
    @Passed
    public static native int center_quadratic_dual(long stream_address,
            long dX1_address, long dX2_address,
            float k11, float k12, float k22,
            float k1, float k2, float C,
            int dim0, int dim1, int dim2,
            long dV_address, long dY_address,
            int mem_width, int mem_stride,
            int partNum);

    /**
     * <pre>
     * (1) deltaX1 = (2*k11) * (deltaY * X1) + k12 * (deltaY * X2) + k1*deltaY
     * (2) deltaX2 = (2*k22) * (deltaY * X2) + k12 * (deltaY * X1) + k2*deltaY.
     *
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * for [height, width] from [100, 40] to(+1, +1) [105, 64]: correct
     * [height, width] = [1024, 1024]: Time = 0.102000, Speed = 38.296566 GB/s
     * </pre>
     *
     * @param stream_address
     * @param d_deltaY_address
     * @param dX1_address
     * @param dX2_address
     * @param k11
     * @param k12
     * @param k22
     * @param k1
     * @param k2
     * @param C
     * @param dim0
     * @param dim1
     * @param dim2
     * @param d_deltaX1_address
     * @param dV_address
     * @param d_deltaX2_address
     * @param mem_width
     * @param mem_stride
     * @param partNum
     * @return
     */
    @Passed
    public static native int center_quadratic_dual_deltaX(long stream_address,
            long d_deltaY_address,
            long dX1_address, long dX2_address,
            float k11, float k12, float k22,
            float k1, float k2, float C,
            int dim0, int dim1, int dim2,
            long d_deltaX1_address,//result0
            long dV_address, long d_deltaX2_address,//result1
            int mem_width, int mem_stride,
            int partNum);
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="row reduce">
    //<editor-fold defaultstate="collapsed" desc="row linear">
    /**
     * <pre>
     * (1) stride = (width + 3)/4 * 4
     * (2) N = field_length
     * (3) M = row_lengthv(the lengthv of the reshaped row vector, which
     *  contains the result of this reduction)
     * (4) N*M = X.lengthv
     * (5) M % stride == 0
     * (6) reshape: X -> Tensor[N, M]
     *  find binomial summary of each row vector of X:
     *      for i from 1 to N:
     *          Y[i] = sum(alpha*X[i] + beta, 0, M).
     *
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * for [height, width] from [100, 40] to(+1, +1) [105, 64]: correct
     * [height, width] = [1024, 1024]:  Time = 0.107000, Speed = 36.507008 GB/s
     * </pre>
     *
     * @param cudaStream_address
     * @param dX_address
     * @param alpha
     * @param beta
     * @param N
     * @param M
     * @param dV_address
     * @param dY_address
     * @param mem_width
     * @param mem_stride
     * @param partNum
     * @return
     */
    @Passed
    public static native int row_linear(long cudaStream_address,
            long dX_address,
            float alpha, float beta,
            int N, int M,
            long dV_address, long dY_address,
            int mem_width, int mem_stride,
            int partNum);

    /**
     * <pre>
     * (1) stride = (width + 3)/4 * 4
     * (2) N = field_length
     * (3) M = row_lengthv(the lengthv of the reshaped row vector, which
     *  contains the result of this reduction)
     * (4) N*M = X.lengthv
     * (5) M % stride == 0
     * (6) reshape: X -> Tensor[N, M]
     *  find binomial summary of each row vector of X:
     *      for i from 1 to N:
     *          Y[i] = sum(alpha*X1[i] + beta*X2[i] + gamma, 0, M).
     *
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * for [height, width] from [100, 40] to(+1, +1) [105, 64]: correct
     * [height, width] = [1024, 1024]:  Time = 0.107000, Speed = 36.507008 GB/s
     * </pre>
     *
     * @param cudaStream_address
     * @param dX1_address
     * @param dX2_address
     * @param alpha
     * @param beta
     * @param gamma
     * @param N
     * @param M
     * @param dV_address
     * @param dY_address
     * @param mem_width
     * @param mem_stride
     * @param partNum
     * @return
     */
    @Passed
    public static native int row_linear_dual(long cudaStream_address,
            long dX1_address, long dX2_address,
            float alpha, float beta, float gamma,
            int N, int M,
            long dV_address, long dY_address,
            int mem_width, int mem_stride,
            int partNum);

    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="row quadratic">
    /**
     * <pre>
     * (1) stride = (width + 3)/4 * 4
     * (2) N = field_length
     * (3) M = row_lengthv(the lengthv of the reshaped row vector, which
     *  contains the result of this reduction)
     * (4) N*M = X.lengthv
     * (5) M % stride == 0
     * (6) reshape: X -> Tensor[N, M]
     *  find binomial summary of each row vector of X:
     *      for i from 1 to N:
     *          Y[i] = sum(alpha*X[i] + beta*X[i] + gamma, 0, M).
     *
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * for [height, width] from [100, 40] to(+1, +1) [105, 64]: correct
     * [height, width] = [1024, 1024]:  Time = 0.107000, Speed = 36.507008 GB/s
     * </pre>
     *
     * @param cudaStream_address
     * @param dX_address
     * @param alpha
     * @param beta
     * @param gamma
     * @param N
     * @param M
     * @param dV_address
     * @param dY_address
     * @param mem_width
     * @param mem_stride
     * @param partNum
     * @return
     */
    @Passed
    public static native int row_quadratic(long cudaStream_address,
            long dX_address,
            float alpha, float beta, float gamma,
            int N, int M,
            long dV_address, long dY_address,
            int mem_width, int mem_stride,
            int partNum);

    /**
     * <pre>
     * (1) stride = (width + 3)/4 * 4
     * (2) N = field_length
     * (3) M = row_lengthv(the lengthv of the reshaped row vector, which
     *  contains the result of this reduction)
     * (4) N*M = X.lengthv
     * (5) M % stride == 0
     * (6) reshape: [X1, X2] -> Tensor[N, M]
     *  find binomial summary of each row vector of X:
     *      for i from 1 to N:
     *          Y[i] = sum(k11*X1[i]^2 + k12*X1[i]*X2[i] + k22*X2[i]*X2[i]
     *                      + k1*X1[i] + k2*X2[i] + C, 0, M).
     *
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * for [height, width] from [100, 40] to(+1, +1) [105, 64]: correct
     * [height, width] = [1024, 1024]: Time = 0.102000, Speed = 38.296566 GB/s
     * </pre>
     *
     * @param cudaStream_address
     * @param dX1_address
     * @param dX2_address
     * @param N
     * @param k11
     * @param M
     * @param k12
     * @param dV_address
     * @param k22
     * @param dY_address
     * @param k1
     * @param k2
     * @param C
     * @param mem_width
     * @param mem_stride
     * @param partNum
     * @return
     */
    @Passed
    public static native int row_quadratic_dual(long cudaStream_address,
            long dX1_address, long dX2_address,
            float k11, float k12, float k22,
            float k1, float k2, float C,
            int N, int M,
            long dV_address, long dY_address,
            int mem_width, int mem_stride,
            int partNum);

    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="row linear_quadratic">
    /**
     * <pre>
     * (1) stride = (width + 3)/4 * 4
     * (2) N = field_length
     * (3) M = row_lengthv(the lengthv of the reshaped row vector, which
     *  contains the result of this reduction)
     * (4) N*M = X.lengthv
     * (5) M % stride == 0
     * (6) reshape: X -> Tensor[N, M]
     *  find linear\quadratic summary of each row vector of X:
     *      for i from 1 to N:
     *          Y1[i] = sum(alpha*X[i] + beta, 0, M)
     *          Y2[i] = sum(alpha*X[i]^2 + beta*X[i] + gamma, 0, M).
     *
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * for [height, width] from [100, 40] to(+1, +1) [105, 64]: correct
     * [height, width] = [1024, 1024]:  Time = 0.107000, Speed = 36.507008 GB/s
     * </pre>
     *
     * @param stream1
     * @param stream2
     * @param dX_address
     * @param alpha1
     * @param beta1
     * @param alpha2
     * @param beta2
     * @param N
     * @param gamma2
     * @param M
     * @param dV1_address
     * @param dY1_address
     * @param dV2_address
     * @param dY2_address
     * @param mem_width
     * @param mem_stride
     * @param partNum
     * @return
     */
    @Passed
    public static native int row_linear_quadratic(long stream1, long stream2,
            long dX_address,
            float alpha1, float beta1,
            float alpha2, float beta2, float gamma2,
            int N, int M,
            long dV1_address, long dY1_address,
            long dV2_address, long dY2_address,
            int mem_width, int mem_stride,
            int partNum);

    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="row linear_quadratic">
    /**
     * <pre>
     * (1) stride = (width + 3)/4 * 4
     * (2) N = field_length
     * (3) M = row_lengthv(the lengthv of the reshaped row vector, which
     *  contains the result of this reduction)
     * (4) N*M = X.lengthv
     * (5) M % stride == 0
     * (6) reshape: X -> Tensor[N, M]
     *  find linear\quadratic summary of each row vector of X:
     *      for i from 1 to N:
     *          Y[i] = sum(C * (alpha*X1[i] + beta*X2 + gamma)^2 + beta1*X, 0, M).
     *
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * for [height, width] from [100, 40] to(+1, +1) [105, 64]: correct
     * [height, width] = [1024, 1024]:  Time = 0.107000, Speed = 36.507008 GB/s
     * </pre>
     *
     * @param stream_address
     * @param dX1_address
     * @param alpha
     * @param C
     * @param dX2_address
     * @param beta
     * @param gamma
     * @param N
     * @param M
     * @param dV_address
     * @param dY_address
     * @param mem_width
     * @param mem_stride
     * @param partNum
     * @return
     */
    @Passed
    public static native int row_linear2_square_row(long stream_address,
            long dX1_address, long dX2_address,
            float C, float alpha, float beta, float gamma,
            int N, int M,
            long dV_address, long dY_address,
            int mem_width, int mem_stride,
            int partNum);
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="row max, min">
    /**
     * <pre>
     * (1) stride = (width + 3)/4 * 4
     * (2) N = field_length
     * (3) M = row_lengthv(the lengthv of the reshaped row vector, which
     *  contains the result of this reduction)
     * (4) N*M = X.lengthv
     * (5) M % stride == 0
     * (6) reshape: X -> Tensor[N, M]
     *  find binomial summary of each row vector of X:
     *      for i from 1 to N:
     *          Y[i] = max(alpha*X[i] + beta*X[i] + gamma, 0, M).
     *
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * for [height, width] from [100, 40] to(+1, +1) [105, 64]: correct
     * [height, width] = [1024, 1024]: Time = 0.115000, Speed = 33.967388 GB/s
     * </pre>
     *
     * @param cudaStream_address
     * @param dX_address
     * @param N
     * @param M
     * @param dV_address
     * @param dY_address
     * @param mem_width
     * @param mem_stride
     * @param partNum
     * @return
     */
    @Passed
    public static native int row_max(long cudaStream_address,
            long dX_address,
            int N, int M,
            long dV_address, long dY_address,
            int mem_width, int mem_stride,
            int partNum);

    /**
     * <pre>
     * (1) stride = (width + 3)/4 * 4
     * (2) N = field_length
     * (3) M = row_lengthv(the lengthv of the reshaped row vector, which
     *  contains the result of this reduction)
     * (4) N*M = X.lengthv
     * (5) M % stride == 0
     * (6) reshape: X -> Tensor[N, M]
     *  find binomial summary of each row vector of X:
     *      for i from 1 to N:
     *          Y[i] = min(alpha*X[i] + beta*X[i] + gamma, 0, M).
     *
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * for [height, width] from [100, 40] to(+1, +1) [105, 64]: correct
     * [height, width] = [1024, 1024]:  Time = 0.108000, Speed = 36.168980 GB/s
     * </pre>
     *
     * @param cudaStream_address
     * @param dX_address
     * @param N
     * @param M
     * @param dV_address
     * @param dY_address
     * @param mem_width
     * @param mem_stride
     * @param partNum
     * @return
     */
    @Passed
    public static native int row_min(long cudaStream_address,
            long dX_address,
            int N, int M,
            long dV_address, long dY_address,
            int mem_width, int mem_stride,
            int partNum);

    /**
     * <pre>
     * (1) stride = (width + 3)/4 * 4
     * (2) N = field_length
     * (3) M = row_lengthv(the lengthv of the reshaped row vector, which
     *  contains the result of this reduction)
     * (4) N*M = X.lengthv
     * (5) M % stride == 0
     * (6) reshape: X -> Tensor[N, M]
     *  find binomial summary of each row vector of X:
     *      for i from 1 to N:
     *          Y[i] = max(alpha*X[i] + beta*X[i] + gamma, 0, M).
     *
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * for [height, width] from [100, 40] to(+1, +1) [105, 64]: correct
     * [height, width] = [1024, 1024]: Time = 0.115000, Speed = 33.967388 GB/s
     * </pre>
     *
     * @param cudaStream_address
     * @param dX_address
     * @param N
     * @param M
     * @param dV_address
     * @param dVIndex_address
     * @param dY_address
     * @param mem_width
     * @param dIndex_address
     * @param mem_stride
     * @param partNum
     * @return
     */
    @Passed
    public static native int row_max_indexed(long cudaStream_address,
            long dX_address,
            int N, int M,
            long dV_address, long dVIndex_address,
            long dY_address, long dIndex_address,
            int mem_width, int mem_stride,
            int partNum);

    /**
     * <pre>
     * (1) stride = (width + 3)/4 * 4
     * (2) N = field_length
     * (3) M = row_lengthv(the lengthv of the reshaped row vector, which
     *  contains the result of this reduction)
     * (4) N*M = X.lengthv
     * (5) M % stride == 0
     * (6) reshape: X -> Tensor[N, M]
     *  find binomial summary of each row vector of X:
     *      for i from 1 to N:
     *          Y[i] = min(alpha*X[i] + beta*X[i] + gamma, 0, M).
     *
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * for [height, width] from [100, 40] to(+1, +1) [105, 64]: correct
     * [height, width] = [1024, 1024]:  Time = 0.108000, Speed = 36.168980 GB/s
     * </pre>
     *
     * @param cudaStream_address
     * @param dX_address
     * @param N
     * @param M
     * @param dV_address
     * @param dVIndex_address
     * @param dY_address
     * @param dIndex_address
     * @param mem_width
     * @param mem_stride
     * @param partNum
     * @return
     */
    @Passed
    public static native int row_min_indexed(long cudaStream_address,
            long dX_address,
            int N, int M,
            long dV_address, long dVIndex_address,
            long dY_address, long dIndex_address,
            int mem_width, int mem_stride,
            int partNum);

    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="row_softmax">
    /**
     * <pre>
     * (1) stride = (width + 3)/4 * 4
     * (2) N = field_length
     * (3) M = row_lengthv(the lengthv of the reshaped row vector, which
     *  contains the result of this reduction)
     * (4) N*M = X.lengthv
     * (5) M % stride == 0
     * (6) reshape: X -> Tensor[N, M]
     * (7) SoftMax: Y[i,j] =  exp(X[i,j])/ sum(exp(A[i], 0, M).
     *
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * for [height, width] from [100, 40] to(+1, +1) [105, 64]: correct
     * [height, width] = [1024, 1024]: Time = 0.290000, Speed = 26.939655 GB/s
     * </pre>
     *
     * @param cudaStream_address
     * @param dX_address
     * @param d_maxA_address
     * @param d_expX_address
     * @param N
     * @param M
     * @param dV_address
     * @param mem_width
     * @param mem_stride
     * @param partNum
     * @return
     */
    @Passed
    public static native int row_softmax(long cudaStream_address,
            long dX_address, long d_maxA_address,
            long d_expX_address,
            int N, int M,
            long dV_address,
            int mem_width, int mem_stride,
            int partNum);

    /**
     * <pre>
     * (1) stride = (width + 3)/4 * 4
     * (2) N = field_length
     * (3) M = row_lengthv(the lengthv of the reshaped row vector, which
     *  contains the result of this reduction)
     * (4) N*M = X.lengthv
     * (5) M % stride == 0
     * (6) reshape: X -> Tensor[N, M]
     * (7) SoftMax: Y[i,j] =  exp(X[i,j])/ sum(exp(A[i], 0, M).
     *
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * for [height, width] from [100, 40] to(+1, +1) [105, 64]: correct
     * [height, width] = [1024, 1024]: Time = 0.290000, Speed = 26.939655 GB/s
     * </pre>
     *
     * @param cudaStream_address
     * @param dX_address
     * @param d_maxA_address
     * @param N
     * @param M
     * @param dV_address
     * @param mem_width
     * @param mem_stride
     * @param partNum
     * @return
     */
    @Passed
    public static native int row_softmaxCrossEntropy_stage1(long cudaStream_address,
            long dX_address, long d_maxA_address,
            int N, int M,
            long dV_address,
            int mem_width, int mem_stride,
            int partNum);
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="layerNorm.deltaXp: v1 & v2">
    /**
     * <pre>
     * (1) stride = (width + 3)/4 * 4
     * (2) N = field_length
     * (3) M = row_lengthv(the lengthv of the reshaped row vector, which
     *  contains the result of this reduction)
     * (4) N*M = X.lengthv
     * (5) M % stride == 0
     * (6) reshape: X -> Tensor[N, M]
     * (7) deltaXp1[N] = row_sum: deltaY * (Y*X_mean - X_std)
     * (8) deltaXp2[N] = row_sum: deltaY * Y.
     * [1] V1: holdY(), Y is not changed
     * [2] affine = false.
     * </pre>
     *
     * @param stream1
     * @param stream2
     * @param d_deltaY_address
     * @param dY_address
     * @param dX_mean_address
     * @param dX_square_mean_address
     * @param eps
     * @param N
     * @param M
     * @param d_deltaXp1_address
     * @param d_deltaXp2_address
     * @param mem_width
     * @param mem_stride
     * @param partNum
     * @return
     */
    @Passed
    public static native int row_layernorm_deltaXp_v1(long stream1, long stream2,
            long d_deltaY_address, long dY_address,
            long dX_mean_address,
            long dX_square_mean_address, float eps,
            int N, int M,
            long d_deltaXp1_address, long d_deltaXp2_address,
            int mem_width, int mem_stride,
            int partNum);

    /**
     * <pre>
     * (1) stride = (width + 3)/4 * 4
     * (2) N = field_length
     * (3) M = row_lengthv(the lengthv of the reshaped row vector, which
     *  contains the result of this reduction)
     * (4) N*M = X.lengthv
     * (5) M % stride == 0
     * (6) reshape: X -> Tensor[N, M]
     * (7) deltaXp1[N] = row_sum: deltaY * { (Y - B)*X_mean - A*X_std }
     * (8) deltaXp2[N] = row_sum: deltaY * (Y - B).
     * [1] V1: holdY(), Y is not changed
     * [2] affine = true.
     * </pre>
     *
     * @param stream1
     * @param stream2
     * @param d_deltaY_address
     * @param dY_address
     * @param dX_mean_address
     * @param dX_square_mean_address
     * @param eps
     * @param dA_address
     * @param dB_address
     * @param N
     * @param M
     * @param d_deltaXp1_address
     * @param d_deltaXp2_address
     * @param mem_width
     * @param mem_stride
     * @param partNum
     * @return
     */
    @Passed
    public static native int row_layernorm_affined_deltaXp_v1(long stream1, long stream2,
            long d_deltaY_address, long dY_address,
            long dX_mean_address,
            long dX_square_mean_address, float eps,
            long dA_address, long dB_address,
            int N, int M,
            long d_deltaXp1_address, long d_deltaXp2_address,
            int mem_width, int mem_stride,
            int partNum);

    /**
     * <pre>
     * (1) stride = (width + 3)/4 * 4
     * (2) N = field_length
     * (3) M = row_lengthv(the lengthv of the reshaped row vector, which
     *  contains the result of this reduction)
     * (4) N*M = X.lengthv
     * (5) M % stride == 0
     * (6) reshape: X -> Tensor[N, M]
     * (7) deltaXp1[N] = row_sum: deltaY * (X*X_mean - X_squareMean - eps)
     * (8) deltaXp2[N] =  row_sum: deltaY * (X - X_mean).
     * [1] V2: holdX(), X is not changed
     * [2] affine = false
     * </pre>
     *
     * @param stream1
     * @param stream2
     * @param d_deltaY_address
     * @param dX_address
     * @param dX_mean_address
     * @param dX_square_mean_address
     * @param eps
     * @param N
     * @param M
     * @param d_deltaXp1_address
     * @param d_deltaXp2_address
     * @param mem_width
     * @param mem_stride
     * @param partNum
     * @return
     */
    @Passed
    public static native int row_layernorm_deltaXp_v2(long stream1, long stream2,
            long d_deltaY_address, long dX_address,
            long dX_mean_address,
            long dX_square_mean_address, float eps,
            int N, int M,
            long d_deltaXp1_address, long d_deltaXp2_address,
            int mem_width, int mem_stride,
            int partNum);

    /**
     * <pre>
     * (1) stride = (width + 3)/4 * 4
     * (2) N = field_length
     * (3) M = row_lengthv(the lengthv of the reshaped row vector, which
     *  contains the result of this reduction)
     * (4) N*M = X.lengthv
     * (5) M % stride == 0
     * (6) reshape: X -> Tensor[N, M]
     * (7) deltaXp1[N] = row_sum: deltaY * (X*X_mean - X_squareMean - eps)
     * (8) deltaXp2[N] =  row_sum: deltaY * (X - X_mean).
     * [1] V2: holdX(), X is not changed
     * [2] affine = true.
     * </pre>
     *
     * @param stream1
     * @param stream2
     * @param d_deltaY_address
     * @param dX_address
     * @param dX_mean_address
     * @param dX_square_mean_address
     * @param eps
     * @param dA_address
     * @param N
     * @param M
     * @param d_deltaXp1_address
     * @param d_deltaXp2_address
     * @param mem_width
     * @param mem_stride
     * @param partNum
     * @return
     */
    @Passed
    public static native int row_layernorm_affined_deltaXp_v2(long stream1, long stream2,
            long d_deltaY_address, long dX_address,
            long dX_mean_address,
            long dX_square_mean_address, float eps,
            long dA_address,
            int N, int M,
            long d_deltaXp1_address, long d_deltaXp2_address,
            int mem_width, int mem_stride,
            int partNum);
    //</editor-fold>
    //</editor-fold>
}
