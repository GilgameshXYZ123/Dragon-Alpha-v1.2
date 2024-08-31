/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.util.math;

import static java.lang.Math.PI;
import z.util.function.Function_float64;
import z.util.lang.exception.IAE;
import z.util.lang.annotation.Passed;

/**
 *
 * @author dell
 */
public final class ExMath 
{         
    private ExMath() {}
    
    //<editor-fold defaultstate="collapsed" desc="class: Function">
    public static interface HashCoder {   
        public int hashCode(Object key);
    }
    
    public static final HashCoder DEF_HASHCODER = new HashCoder() {
        @Override
        public int hashCode(Object key) {
            if(key == null) return 0;
            int h = key.hashCode();
            return ((h = h^(h >>> 16)) > 0? h : -h);
        }
    };
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="float32 functions">
    public static float sinf(float x) { return FP32Math.sinf(x); }
    public static float cosf(float x) { return FP32Math.cosf(x); }
    public static float tanf(float x) { return FP32Math.tanf(x); }
    
    public static float asinf(float x) { return FP32Math.asinf(x); }
    public static float acosf(float x) { return FP32Math.acosf(x); }
    public static float atanf(float x) { return FP32Math.atanf(x); }
    public static float atan2f(float y, float x) { return FP32Math.atan2f(y, x); }
    
    public static float sinhf(float x) { return FP32Math.sinhf(x); }
    public static float coshf(float x) { return FP32Math.coshf(x); }
    public static float tanhf(float x) { return FP32Math.tanhf(x); }
    
    public static float expf(float x) { return FP32Math.expf(x); }
    public static float expm1f(float x) { return FP32Math.expm1f(x); }
    public static float powf(float x, float y) { return FP32Math.powf(x, y); }
    
    public static final float RLOG2_F32 = (float) (1.0 / Math.log(2));
    public static float logf(float x) { return FP32Math.logf(x); }
    public static float log1pf(float x) { return FP32Math.log1pf(x); }
    public static float log2f(float x){ return logf(x) * RLOG2_F32; }
    
    public static float sqrtf(float x) { return FP32Math.sqrtf(x); }
    public static float cbrtf(float x) { return FP32Math.cbrtf(x); }
    public static float hypotf(float x, float y) { return FP32Math.hypotf(x, y); }
    
    public static float ceilf(float x) { return FP32Math.ceilf(x); }
    public static float floorf(float x) { return FP32Math.floorf(x); }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="basic functions">
    public static double sin(double a) { return Math.sin(a); }
    public static double cos(double a) { return Math.cos(a); }
    public static double tan(double a) { return Math.tan(a); }

    public static double asin(double a) { return Math.asin(a); }
    public static double acos(double a) { return Math.acos(a); }
    public static double atan(double a) { return Math.atan(a); }
    public static double atan2(double y, double x) { return Math.atan2(y, x); }
    
    public static double exp(double a) { return Math.exp(a); }
    public static double pow(double a, double b) { return Math.pow(a, b); }
    
    public static final double RLOG2_F64 = 1.0 / Math.log(2);
    public static double log(double a) { return Math.log(a); }
    public static double log10(double a) { return Math.log10(a); }
    public static double log2(double x){ return Math.log(x) * RLOG2_F64; }
      
    public static double sqrt(double a) { return Math.sqrt(a); }
    public static double cbrt(double a) { return Math.cbrt(a); }
     
    public static double ceil(double a) { return Math.ceil(a);  }
    public static double floor(double a) { return Math.floor(a); }
    
    public static boolean is_nan(float x) { return x != x; }
    public static boolean is_inf(float x) { return x == x && is_nan(x - x);  }
    
    public static double clip(double v, double min, double max) { if(v < min) v = min; if(v > max) v = max; return v; }
    public static float  clip(float  v, float  min, float  max) { if(v < min) v = min; if(v > max) v = max; return v; }
    public static int    clip(int    v, int    min, int    max) { if(v < min) v = min; if(v > max) v = max; return v; }
    public static short  clip(short  v, short  min, short  max) { if(v < min) v = min; if(v > max) v = max; return v; }
    public static char   clip(char   v, char   min, char   max) { if(v < min) v = min; if(v > max) v = max; return v; }
    public static byte   clip(byte   v, byte   min, byte   max) { if(v < min) v = min; if(v > max) v = max; return v; }
    
    public static boolean isPerfectSqurae(int v) {
        int r = v;
        while (r*r > v) r = (r + v/r) >> 1;// Xn+1 = Xn - f(Xn)/f'(Xn)
        return r*r == v;
    }
    
    public static boolean is_prime(int v) {
        if(v < 0) v = -v;
        if(v <= 1) return false;
        for(int i = 2, len = (int)sqrt(v); i <= len; i++)
            if(v % i == 0) return false;
        return true;
    }
    
    /**
     * <pre>
     * get the value of a specific index {@code n} of Fibonacci Sequence,
     * while the index of sequence is from 0;
     * 1,1,2,3,5,8,13,21,44,65......
     * </pre>
     * @param n
     * @return 
     */
    public static long fibonacci(int n) {
        if(n < 0) throw new IAE("The index of the sequence mustn't be negative"); 
        if(n <= 1) return 1;
        int a = 1,b = 1,c;
        for (int i = 2; i <= n; i++) { c = a + b; a = b; b = c; }
        return b;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Gaussian-Distribute">
    //<editor-fold defaultstate="collapsed" desc="class NDdensity implements Fuction">
    public static final class GaussianDensity implements Function_float64 {
        private final double mu;
        private final double div;
        private final double k;
        
        public GaussianDensity() { this(0,1); }
        public GaussianDensity(double avg, double sigma) {
            this.mu = avg;
            this.div = -2 * sigma * sigma;
            this.k = 1.0 / (sqrt(2 * PI) * sigma);
        }
        
        @Override public double apply(double x) { x -= mu; return k * exp(x * x / div); }
    }
    private static final GaussianDensity N_1_0=new GaussianDensity();
    //</editor-fold>
    /**
     * <pre>
     * This function is used to approximate the standard normal distribution,
     * with the min=0 and variance=1.
     * we regard the the densitiy function of Normal Distribute(NF) as f(x):
     * compute:
     * (1)I1=the integral of f(x) in(-infinite, x);
     * (2)I2=the integral of f(x) in (-x,x);
     * Theoritically, I1=(I2+1)/2, but practically, there exists samll difference
     * between I1 and I2, so we return (I1+I2)/2 to improve the presition.
     * </pre>
     * @param x
     * @return 
     */
    @Passed
    public static double gaussianDistribute(double x)
    {
        if(x<1e-9&&x>-1e-9) return 0.5;
        double r1=(ExMath.integral(-x, x, N_1_0, 1e-7)+1)/2;
        double r2=ExMath.integralFromNF(x, N_1_0);
        return (r1+r2)/2;
    }
    /**
     * <pre>
     * This function is used to approximate the standard normal distribution,
     * with the min=0 and variance=1.
     * we regard the the densitiy function of Normal Distribute(NF) as f(x):
     * compute the integral for f(x) in(start,end);
     * </pre>
     * @param start
     * @param end
     * @return 
     */
    @Passed
    public static double gaussianDistribute(double start, double end)
    {
        return ExMath.integral(start, end, N_1_0, 1e-7);
    }
    /**
     * By equivalent transformation, we use standard NF to get the value
     * of NF with a specific avg and stddev.
     * @see #gaussianDistribute(double) 
     * @param x
     * @param avg
     * @param stddev
     * @return 
     */
    @Passed
    public static double gaussianDistribute(double x, double avg, double stddev)
    {
        return ExMath.gaussianDistribute((x-avg)/stddev)/stddev;
    }
    /**
     * By equivalent transformation, we use standard NF to get the value
     * of NF with a specific avg and stddev.
     * @see #gaussianDistribute(double, double) 
     * @param start
     * @param end
     * @param avg
     * @param stddev
     * @return 
     */
    @Passed
    public static double gaussianDistribute(double start, double end, double avg, double stddev)
    {
        return ExMath.gaussianDistribute((start-avg)/stddev, (end-avg)/stddev)/stddev;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Narrow-Function">
    //<editor-fold defaultstate="collapsed" desc="Inner-Code">
    private static final double[] exp_k = ExMath.create_exp();
    private static final double[] sinh_k = ExMath.create_sinh();
    private static final double[] cosh_k = ExMath.create_cosh();
    
    private static double[] create_exp() {//exp(x)
        double k[] = new double[12], c = 1;
        for (int i=0,j=0;i<k.length;i++) { c /= ++j; k[i] = c;}
        return k;
    }

    private static double[] create_sinh() {//(exp(x)-exp(-x))/2
        double k[] = new double[12], c = 1; k[0] = c;
        for (int i = 1, j = 1; i < k.length; i++) { c /= ++j * ++j; k[i] = c; }
        return k;
    }
    
    private static double[] create_cosh() {
        double k[] = new double[12], c = 0.5; k[0] = c;
        for (int i = 1, j = 2; i < k.length; i++) { c /= ++j * ++j; k[i] = c; }
        return k;
    }
    //</editor-fold>
    public static double __exp(double x) {
        double y = 1,cur = x;
        for (int i=0; i<exp_k.length; i++) { y += cur * exp_k[i]; cur *= x; }
        return y;
    }

    public static double __sinh(double x) {
        double y = 0, cur = x; x *= x;
        for (int i = 0; i < sinh_k.length; i++) { y += cur * sinh_k[i]; cur *= x; }
        return y;
    }

    public static double __cosh(double x) {
        double y = 1, cur = x *= x;
        for (int i = 0; i < cosh_k.length; i++) { y += cur * cosh_k[i]; cur *= x; }
        return y;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Integral-Function">
    private static final int MIN_INTEGRAL_P = 500000;
    private static final int MAX_INTEGRAL_P = 1000000;
    
    public static final double DEF_INTEGRAL_STEP=1e-7;
    public static final double INTEGRAL_POSITIVE_ZERO=1e-19;
    public static final double INEGRAL_POSITIVE_ZERO_RECIPROCAL=1e4;
    
    //<editor-fold defaultstate="collapsed" desc="Inner-Code:Compute-Integral">
    /**
     * <pre>
     * (1)Compute the integral for {@ Function func} on interval from start
     * to end; 
     * (2)This method use an approximation Algorithm, the precision
     * is based on step, while the number of steps is p;
     * (3)This Method is an lower interface, kindly use, other higher 
     * interface to avoid use this function directly
     * (4)to avoid spend too mush resource, the number of steps is limited
     * on interval [50000,100000];
     * </pre>
     * @param start the Lower bound of this intergral
     * @param step
     * @param p
     * @param func
     * @return 
     */
    @Passed
    private static double avgIntegral(double start, double step, int p, Function_float64 func) {
        double sum = func.apply(start) / 2;
        start += step;
        for (int i = 1; i < p; i++, start += step)  sum += func.apply(start); 
        sum -= func.apply(start) / 2;
        return sum * step;
    }
    
    /**
     * <pre>
     * The integral from start to end {@code Function func}.
     * (1)check the start and the end of integral; sometimes {@code start < end}, 
     * in this case exchange(start, end), and set flag=-1;
     * (2)compute the step and step_number based on the interval and dx
     * (3)for integral from negative infinite or to positive infinite,
     * you need do some transformation on the function and boundary, then 
     * you can use this method, but somtimes there may exist an larger 
     * deviation between result and practial value;
     * </pre>
     * @param start the lower bound of intergral
     * @param end the higher bound of integral
     * @param func
     * @param dx the expected step
     * @throws IAE {@code  if div<dx*100}, the step number is too few for integral
     * @throws IAE {@code if step>1e-6*dx}, the step is too big for intergral
     * @return 
     */
    public static double integral(double start, double end, Function_float64 func, double dx) {
        double step = dx, div = end - start, flag = 1;
        if(div < 0) { flag = start; start = end; end = flag; div *= -1; flag = -1; }
        if(div <= dx * 100) throw new IAE("Need more segment for dx to integral[div<dx*100]");
        
        int p = (int) ((end - start) / dx);
        if (p < MIN_INTEGRAL_P) {  p = MIN_INTEGRAL_P; step = div / MIN_INTEGRAL_P; }
        else if (p > MAX_INTEGRAL_P) {
            p = MAX_INTEGRAL_P;
            step = div / MAX_INTEGRAL_P;
            if (step > dx * 1e6) throw new IAE("Need smaller step to integral[step>dx*1e6]");
        }
        return flag * ExMath.avgIntegral(start, step, p, func);
    }
    /**
     * <pre>
 The integral from +infinite to base for Function func.
     * Divide the interval to two parts: [greater than 0], [less than 0],
     * as base is not an infinite value;
     * (1)for integral (0, +infinite) do some transimition and compute
     *  =>fx->1/(x*x)*f(1/x)
     * (2)for integral [-base, 0), just compute the integral
     * must avoid the integral at 0, so in effect the integral algorithm
     * is like this:
     * {@code if(base>0) return ExMath.integral(1e-19, end, funcs, dx);
     *   else return ExMath.integral(base, 0, func, dx)
     *           +ExMath.integral(1e-19, 1e3, func, dx);}
     * </pre>
     * @param base
     * @param func
     * @param dx
     * @return 
     */
    @Passed
    public static double integralToPF(double base, Function_float64 func,double dx)
    {
        if(base==0) throw new IAE("bound can not equal to zero");
        double end=1/base;
        Function_float64 funcs = (double x) -> (x=1/x)*func.apply(x)*x;
        if(base>0) return ExMath.integral(INTEGRAL_POSITIVE_ZERO, end, funcs, dx);
        else return ExMath.integral(base, 0, func, dx)
                +ExMath.integral(INTEGRAL_POSITIVE_ZERO, INEGRAL_POSITIVE_ZERO_RECIPROCAL, funcs, dx);
    }
    /**
     * <pre>
     * The integral from -infinite to bound for {@code Function func}.
     * (1)do such a transimition:
     * {@code ExMath.integral(-infinite, bound, function(x))
     *          =ExMath.integral(-bound, +infinite, function(-x))},as:
     * {@code ExMath.integralFromNF(bound, funcion(x))
     *          =ExMath.integralFromPF(-bound, function(-x));}
     * 
     * </pre>
     * @param bound
     * @param func
     * @param dx
     * @return 
     */
    @Passed
    public static double integralFromNF(double bound, Function_float64 func, double dx)
    {
        if(bound==0) throw new IAE("bound can not equal to zero");
        double end=-1/bound;
        Function_float64 funcs=(double x) -> (x=1/-x)*func.apply(x)*x;
        if(bound<0) return ExMath.integral(INTEGRAL_POSITIVE_ZERO, end, funcs, dx);
        else return ExMath.integral(-bound, 0, func, dx)
                +ExMath.integral(INTEGRAL_POSITIVE_ZERO, INEGRAL_POSITIVE_ZERO_RECIPROCAL, funcs, dx);
    }
    //</editor-fold>
    public static double integral(double start, double end, Function_float64 func) { return ExMath.integral(start, end, func, DEF_INTEGRAL_STEP); }
    public static double integralToPF(double start, Function_float64 func) { return ExMath.integralToPF(start, func, DEF_INTEGRAL_STEP); }
    public static double integralFromNF(double bound, Function_float64 func) { return ExMath.integralFromNF(bound, func, DEF_INTEGRAL_STEP); }
    //<editor-fold defaultstate="collapsed" desc="Inner-Code:Find-Integral-Boundary">
    private static double findAvgIntegralHB(double start, double step, double expect, Function_float64 func) {
        expect /= step;
        double sum = func.apply(start)/2, lastdiv=expect,nextdiv=expect-sum;
        if(nextdiv<0) nextdiv=-nextdiv;
        while(lastdiv>=nextdiv)
        {
            lastdiv=nextdiv;
            sum+=func.apply(start+=step);
            nextdiv=expect-sum;
            if(nextdiv<0) nextdiv=-nextdiv;
        }
        sum-=func.apply(step)/2;
        nextdiv=expect-sum;
        if(nextdiv<0) nextdiv=-nextdiv;
        return (lastdiv<nextdiv? start:start+step);
    }
    
    public static double findIntegralHB(double start, double expect, double precision, Function_float64 func) { 
        if(precision < 0) throw new IAE("The precision used to find the highe boundary of this Integral must positive");
        double ex = (expect >= 0 ? expect : -expect);
        if (ex < precision * 10) throw new IAE("the precision is too big to find the highe boundary of this Integral must positive");
        return ExMath.findAvgIntegralHB(start, precision, expect, func);
    }
    //</editor-fold>
    //</editor-fold>
}
