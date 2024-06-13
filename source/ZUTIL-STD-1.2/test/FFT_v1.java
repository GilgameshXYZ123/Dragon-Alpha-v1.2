/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package X.dconv3DX.FFT;

import z.util.math.vector.Vector;


/**
 *
 * @author Gilgamesh
 */
public class FFT_v1 
{
    //<editor-fold defaultstate="collapsed" desc="fft & ifft">
    public static Complex[] fft(Complex[] x)  {
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
        
        //as: (e^-x) and (e^x) are conjugate
        //no matter:
        //<1> kth = -2 * Math.PI * k / N
        //<2> kth = +2 * Math.PI * k / N
        
        Complex[] y = new Complex[N];
        for(int k=0; k<N2; k++) {//butter fly:
            double kth = -2 * Math.PI * k / N;
            Complex wk = new Complex(Math.cos(kth), Math.sin(kth));//e^kth
            y[k     ] = ye[k].add(wk.mul(yo[k]));//<1> y[j] = ye[j] + w[j]*yo[j]
            y[k + N2] = ye[k].sub(wk.mul(yo[k]));//<2> y[j + (n/2)] = ye[j] - w[j]*odd[j]
        }
        return y;
    }        
    
    public static Complex[] ifft(Complex[] x)
    {
        int N = x.length;
        Complex[] y = new Complex[N];
        
        for(int i=0; i<N; i++) y[i] = x[i].conj();
        
        y = fft(y);
        
        for(int i=0; i<N; i++) y[i] = y[i].conj();
        for(int i=0; i<N; i++) y[i] = y[i].scale(1.0f / N);
        
        return y;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="rfft & irfft">
    public static Complex[] rfft(Complex[] x)  {
        x = fft(x);
        for(int i=0; i<x.length; i++) x[i] = x[i].conj();
        return x;
    }      
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="fft_conv && fft_corr">
    public static Complex[] fft_conv(Complex[] x, Complex[] w) {//linear convolution
        if(x.length < w.length) throw new RuntimeException();
        
        Complex zero = new Complex(0, 0);
        
        Complex[] wp = new Complex[x.length];//padding w the same size as x
        for(int i=0; i<w.length; i++) wp[i] = w[i];
        for(int i=w.length; i<x.length; i++) wp[i] = zero;
        
        x = fft(x);
        wp = fft(wp);
        
        System.out.println("X + " + x.length);
 
        Complex[] y0 = new Complex[x.length];
        for(int i=0; i<y0.length; i++) y0[i] = x[i].mul(wp[i]);
        
        y0 = ifft(y0);
        
        int I = x.length, F = w.length, O = I - F + 1;
        Complex[] y  =new Complex[O];
        int index = y0.length - 1;
        for(int i=O-1; i>=0; i--) y[i] = y0[index--];
        return y;
    }
    
    public static Complex[] fft_corr(Complex[] x, Complex[] w) {//linear convolution
        if(x.length < w.length) throw new RuntimeException();
        
        Complex zero = new Complex(0, 0);
        
        Complex[] wp = new Complex[x.length];//padding w the same size as x
        for(int i=0; i<w.length; i++) wp[i] = w[i];
        for(int i=w.length; i<x.length; i++) wp[i] = zero;
        
        x = fft(x);
        wp = rfft(wp);
 
        Complex[] y0 = new Complex[x.length];
        for(int i=0; i<y0.length; i++) y0[i] = x[i].mul(wp[i]);
        y0 = ifft(y0);
        
        int I = x.length, F = w.length, O = I - F + 1;
        Complex[] y = new Complex[O];
        for(int i=0; i<O; i++) y[i] = y0[i];
        return y;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="conv && corr">
    public static float[] conv(float[] x, float[] w)//stride = 1
    {
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
    
    public static float[] corr(float[] x, float[] w)//stride = 1
    {
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
        System.out.println("y = fft(x): "); show(y);
        
        Complex[] z = ifft(y); zero_small(z);
        System.out.println("z = ifft(y)"); show(z);
    }
    
    public static void test2(int I, int F) 
    {
        Complex[] x = new Complex[I];
        for(int i=0; i<x.length; i++) x[i] = new Complex(Math.random(), 0);
        float[] rx = Complex.reals(x);
        
        Complex[] w = new Complex[F];
        for(int i=0; i<w.length; i++) w[i] = new Complex(Math.random(), 0);
        float[] rw = Complex.reals(w);
        
        //FFT conv---------------------------------------------------------------
        Complex[] y = fft_conv(x, w); 
        float[] ry1 = Complex.reals(y);
     
        //normal conv-----------------------------------------------------------
        float[] ry2 = conv(rx, rw);
        
        System.out.println("y = fft_rconv(x, w)"); Vector.println(ry1);
        System.out.println("y = nor_rconv(x, w)"); Vector.println(ry2);
    }
    
    public static void test3(int I, int F) {
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
        
        System.out.println("y = fft_rconv(x, w)"); Vector.println(ry1);
        System.out.println("y = nor_rconv(x, w)"); Vector.println(ry2);
    }
    
    public static void main(String[] args)
    {
//        test1(8);
//        test2(16, 4);
        test3(16, 4);
        
    }
    
        
}
