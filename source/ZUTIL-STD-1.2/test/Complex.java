/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package X.dconv3DX.FFT;

/**
 *
 * @author Gilgamesh
 */
public class Complex 
{
    public static final Complex zero = new Complex(0, 0);
    
    public static float pi = (float) Math.PI;
    private float re;
    private float im;
    
    public Complex(float real, float img) {
        this.re = real;
        this.im = img;
    }
    
    public Complex(double real, double img) {
        this.re = (float) real;
        this.im = (float) img;
    }
    
    public Complex(Complex cpx) {
        this.re = cpx.re;
        this.im = cpx.im;
    }
    
    //<editor-fold defaultstate="collapsed" desc="Basic-Functions">
    public float real() { return re; }
    
    public float img() { return im; }
    
    public void zero_small() { 
        if(re < 1e-6f) re = 0;
        if(im < 1e-6f) im = 0;
    } 
    
    public float phase() { return (float) Math.atan2(im, re); }
    
    public float mod() { return (float) Math.hypot(re, im);  }
    
    @Override
    public String toString() {
        return "<" + re + " + " + im + " i>";
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="running-area: basic-opt">
    public Complex add(Complex b) { return add(this, b); }
    public Complex sub(Complex b) { return sub(this, b); }
    public Complex mul(Complex b) { return mul(this, b); }
    public Complex div(Complex b) { return div(this, b); }
    
    public static Complex add(Complex a, Complex b) {
        float re = a.re + b.re;
        float im = a.im + b.im;
        return new Complex(re, im);
    }
    
    public static Complex sub(Complex a, Complex b) {
        float re = a.re - b.re;
        float im = a.im - b.im;
        return new Complex(re, im);
    }
    
    public static Complex mul(Complex a, Complex b) {
        float re = a.re * b.re - a.im * b.im;
        float im = a.re * b.im + a.im * b.re;
        return new Complex(re, im);
    }
    
    public static Complex div(Complex x, Complex y) {
        float real = (x.re*y.re + x.im*y.im);
        float img  = (x.im*y.re - x.re*y.im);
        float s2 = y.re*y.re + y.im*y.im;
        return new Complex(real / s2, img / s2);
    }
    
    public Complex conj() { return new Complex(re, -im); }
    
    public Complex scale(float alpha) { return new Complex(re*alpha, im*alpha); }
    
    public Complex scale(float alpha, float beta) {
        return new Complex(re*alpha, im*beta);
        
    }
    //</editor-fold>
    
    public static float[] reals(Complex[] arr) {
        float[] re = new float[arr.length];
        for(int i=0; i<arr.length; i++) re[i] = arr[i].re;
        return re;
    }
    
    public static float[][] reals(Complex[][] mat) {
        int N = mat.length, M = mat[0].length;
        float[][] re = new float[N][M];
        for(int i=0; i<N; i++)
        for(int j=0; j<M; j++)
            re[i][j] = mat[i][j].re;
        return re;
    }
    
    public static float[] imags(Complex[] arr) {
        float[] im = new float[arr.length];
        for(int i=0; i<arr.length; i++) im[i] = arr[i].im;
        return im;
    }
    
    public static Complex[] conj(Complex[] arr) {
        int N = arr.length;
        Complex[] Y = new Complex[N];
        for(int i=0; i<N; i++) Y[i] = arr[i].conj();
        return Y;
    }
    
    public static Complex[][] conj(Complex[][] mat) {
        int N = mat.length, M = mat[0].length;
        Complex[][] y = new Complex[N][M];
        for(int i=0; i<N; i++)
        for(int j=0; j<M; j++) y[i][j] = mat[i][j].conj();
        return y;
    }
    
    public static Complex[] complex(float[] reals) {
        int N = reals.length;
        Complex[] cpx = new Complex[N];
        for(int i=0; i<N; i++) cpx[i] = new Complex(reals[i], 0);
        return cpx;
    }
    
    public static Complex[][] complex(float[][] reals) {
        int N = reals.length, M = reals[0].length;
        Complex[][] cpx = new Complex[N][M];
        for(int i=0; i<N; i++)
        for(int j=0; j<M; j++)
            cpx[i][j] = new Complex(reals[i][j], 0);
        return cpx;
    }
    
    public static Complex[][] mul(Complex[][] X, Complex[][] Y) {
        int N = X.length, M = X[0].length;
        Complex[][] Z = new Complex[N][M];
        for(int i=0; i<N; i++)
        for(int j=0; j<M; j++) Z[i][j] = X[i][j].mul(Y[i][j]);
        return Z;
    }
}