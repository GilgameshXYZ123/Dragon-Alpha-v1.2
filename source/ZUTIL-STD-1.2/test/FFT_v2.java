/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

import z.util.math.vector.Vector;


/**
 *
 * @author Gilgamesh
 */
public class FFT_v2 
{
    //<editor-fold defaultstate="collapsed" desc="fft: core">
    public static int padding_to_2pow(int x) {
        int n = x - 1;
        n |= n >>> 1;
        n |= n >>> 2;
        n |= n >>> 4;
        n |= n >>> 8;
        n |= n >>> 16;
        if(n >=  Integer.MAX_VALUE) throw new RuntimeException();
        return ((n < 0) ? 1 : n + 1);
    }
    
    public static Complex[] __fft(Complex[] x) 
    {
        int N = x.length;
        if(N == 1) return new Complex[] { x[0] };//base case
        if(N % 2 != 0) throw new RuntimeException("N is not a power of 2");
        
        int N2 = N >> 1;
        
        Complex[] even = new Complex[N2];//even term coef{0, 2}
        for(int k=0; k<N2; k++) even[k] = x[k << 1];
        Complex[] ye = fft(even);
        
        Complex[] odd = even;//odd term coef{1, 3}
        for(int k=0; k<N2; k++) odd[k] = x[(k << 1) + 1];
        Complex[] yo = fft(odd);
        
        Complex[] y = new Complex[N];
        for(int k=0; k<N2; k++) {//butter fly:
            double kth = -2 * Math.PI * k / N;
            Complex wk = new Complex(Math.cos(kth), Math.sin(kth));//e^kth
            y[k     ] = ye[k].add(wk.mul(yo[k]));//<1> y[j] = ye[j] + w[j]*yo[j]
            y[k + N2] = ye[k].sub(wk.mul(yo[k]));//<2> y[j + (n/2)] = ye[j] - w[j]*odd[j]
        }
        return y;
    }      
    
    public static Complex[] __fft0(Complex[] x) 
    {
        if(x.length == 1) return new Complex[] { x[0] };//base case
        int N = padding_to_2pow(x.length), N2 = N >> 1;
        
        Complex[] even = new Complex[N2];//even term coef{0, 2}
        for(int k=0; k < N2; k++) { 
            int index = k << 1;
            even[k] = (index < x.length? x[index] : Complex.zero);
        }
        Complex[] ye = __fft(even);
        
        Complex[] odd = even;//odd term coef{1, 3}
        for(int k=0; k<N2; k++) {
            int index = (k << 1) + 1;
            odd[k] = (index < x.length? x[index] : Complex.zero);
        }
        Complex[] yo = __fft(odd);
        
        Complex[] y = new Complex[N];
        for(int k=0; k<N2; k++) {//butter fly:
            double kth = -2 * Math.PI * k / N;
            Complex wk = new Complex(Math.cos(kth), Math.sin(kth));//e^kth
            y[k     ] = ye[k].add(wk.mul(yo[k]));//<1> y[j] = ye[j] + w[j]*yo[j]
            y[k + N2] = ye[k].sub(wk.mul(yo[k]));//<2> y[j + (n/2)] = ye[j] - w[j]*odd[j]
        }
        return y;
    }        
    //</editor-fold>
    
    public static Complex[] fft(Complex[] x) { return __fft0(x); }
    
    public static Complex[] rfft(Complex[] x)  {
        x = fft(x);
        for(int i=0; i<x.length; i++) x[i] = x[i].conj();
        return x;
    }      
    
    public static Complex[] ifft(Complex[] x) {
        int N = x.length;
        Complex[] y = new Complex[N];
        for(int i=0; i<N; i++) y[i] = x[i].conj();
        
        y = fft(y);
        
        float k = 1.0f / N;
        for(int i=0; i<N; i++) y[i] = y[i].conj().scale(k);
        return y;
    }
    
    //<editor-fold defaultstate="collapsed" desc="conv && corr">
    public static float[] conv(float[] x, float[] w) {//stride = 1
        int I = x.length, F = w.length;
        int O = I - F + 1; if(O <= 0) throw new RuntimeException();
        float[] y = new float[O];
        
        for(int i=0; i<O; i++) {
            float v = 0;
            for(int j=0; j<F; j++) v += x[j + i] * w[F - 1 - j];
            y[i] = v;
        }
        return y;
    }
    
    public static float[] corr(float[] x, float[] w) {//stride = 1
        int I = x.length, F = w.length;
        int O = I - F + 1; if(O <= 0) throw new RuntimeException();
        float[] y = new float[O];
        
        for(int i=0; i<O; i++) {
            float v = 0;
            for(int j=0; j<F; j++) v += x[j + i] * w[j];
            y[i] = v;
        }
        return y;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="fft_conv && fft_corr">
    public static Complex[] fft_conv(Complex[] x, Complex[] w) {//linear convolution
        if(x.length < w.length) throw new RuntimeException();
        
        Complex[] wp = new Complex[x.length];//padding w the same size as x
        System.arraycopy(w, 0, wp, 0, w.length);
        for(int i=w.length; i<x.length; i++) wp[i] = Complex.zero;
        
        x  = fft(x);
        wp = fft(wp);
        
        Complex[] y0 = new Complex[x.length];
        for(int i=0; i<y0.length; i++) y0[i] = x[i].mul(wp[i]);
        y0 = ifft(y0);
        
        int I = x.length, F = w.length, O = I - F + 1;//clip y
        Complex[] y = new Complex[O];
        int index = y0.length - 1;
        for(int i = O - 1; i>=0; i--) y[i] = y0[index--];
        return y;
    }
    
    public static Complex[] fft_corr(Complex[] x, Complex[] w) {//linear convolution
        if(x.length < w.length) throw new RuntimeException();
        
        Complex[] wp = new Complex[x.length];//padding w the same size as x
        System.arraycopy(w, 0, wp, 0, w.length);
        for(int i=w.length; i<x.length; i++) wp[i] = Complex.zero;
        
        int I = x.length, F = w.length, out_size = I - F + 1;//cilp y
        
        x = fft(x);
        wp = rfft(wp);
 
        Complex[] y0 = new Complex[x.length];
        for(int i=0; i<y0.length; i++) y0[i] = x[i].mul(wp[i]);
        y0 = ifft(y0);

        Complex[] y  =new Complex[out_size];
        System.arraycopy(y0, 0, y, 0, out_size);
        return y;
    }
    //</editor-fold>
    
    public static void zero_small(Complex[] x) {
        for(int i=0; i<x.length; i++) x[i].zero_small();
    }
    
    public static void show(Complex[] x) {
        for(int i=0; i<x.length; i++) System.out.println(x[i] + " ");
        System.out.println();
    }
    
    public static void test1(int N) {
        Complex[] x = new Complex[N];
        for(int i=0; i<N; i++) x[i] = new Complex(Math.random(), 0);//real number
        
        Complex[] y = fft(x);
        Complex[] z = ifft(y);
        
        float[] rx = Complex.reals(x);
        float[] rz = Complex.reals(z);
        
        float sp = Vector.samePercentAbsolute(rx, rz);
        System.out.println("N = " + N);
        System.out.print("rx = "); Vector.println(rx);
        System.out.print("rz = "); Vector.println(rz);
        System.out.println("sp = " + sp + "\n");
        if(sp < 0.99f) throw new RuntimeException();
    }
    
    public static void test2(int I, int F) 
    {
        System.out.println("I, F = " + I + ":" + F);
        
        Complex[] x = new Complex[I];
        for(int i=0; i<x.length; i++) x[i] = new Complex(Math.random(), 0);
        float[] rx = Complex.reals(x);
        
        Complex[] w = new Complex[F];
        for(int i=0; i<w.length; i++) w[i] = new Complex(Math.random(), 0);
        float[] rw = Complex.reals(w);
        
        //FFT conv---------------------------------------------------------------
        Complex[] y = fft_corr(x, w); 
        float[] ry1 = Complex.reals(y);
     
        //normal conv-----------------------------------------------------------
        float[] ry2 = corr(rx, rw);
        
        //compare---------------------------------------------------------------
        float sp = Vector.samePercentAbsolute(ry1, ry2);
        
        System.out.print("fft_corr(x, w)"); Vector.println(ry1);
        System.out.print("nor_corr(x, w)"); Vector.println(ry2);
        System.out.println("sp = " + sp);
        if(ry1.length != ry2.length) throw new RuntimeException(ry1.length + " : " + ry2.length); 
        if(sp < 0.99f) throw new RuntimeException(); 
    }
    
    public static void main(String[] args)
    {
        //for(int i=1; i<131; i++) test1(i);
        for(int i=8; i<131; i++) test2(i, 5);
    }
}
