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
public class FFT_v3 
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
    
    public static Complex[] fft_corr(Complex[] x, Complex[] w) {//linear convolution
        if(x.length < w.length) throw new RuntimeException();
        
        Complex[] wp = new Complex[x.length];//padding w the same size as x
        System.arraycopy(w, 0, wp, 0, w.length);
        for(int i=w.length; i<x.length; i++) wp[i] = Complex.zero;
        
        int I = x.length, F = w.length;//cilp y
        
        x = fft(x);
        wp = rfft(wp);
 
        Complex[] y0 = new Complex[x.length];
        for(int i=0; i<y0.length; i++) y0[i] = x[i].mul(wp[i]);
        return ifft(y0);
    }
    
    public static void fft_conv3D_v1(
            float[][][][] X, int IH, int IW,//[N, IH, IW, IC]
            float[][][][] W, int FH, int FW,//[OC, FH, FW, IC]
            float[][][][] Y, int OH, int OW,//[N, OH, OW, OC]
            int N, int IC, int OC,
            int ph, int pw)
    {
        int IHp = IH + 2*ph;
        int IWp = IW + 2*pw;
        
        Complex[][][] A = new Complex[N][IC][IHp * IWp];//padding A
        for(int n=0; n<N; n++) 
        for(int ic=0; ic<IC; ic++) 
        {
            int index = 0;
            for(int ih=-ph; ih < (IH + ph); ih++)
            for(int iw=-pw; iw < (IW + pw); iw++) 
            {
                boolean lx = (ih>=0) && (ih<IH) && (iw>=0) && (iw<IW);
                A[n][ic][index++] = (lx? 
                        new Complex(X[n][ih][iw][ic], 0) : 
                        Complex.zero);
            }
        }
        
        Complex[][][] B = new Complex[OC][IC][FH * FW];//padding Bs
        for(int oc=0; oc<OC; oc++) 
        for(int ic=0; ic<IC; ic++)
        {
            int index = 0;
            for(int fh=0; fh<FH; fh++)
            for(int fw=0; fw<FW; fw++) {
                B[oc][ic][index++] = new Complex(W[oc][fh][fw][ic], 0);
            }
        }
        
        for(int n=0; n<N; n++)
        for(int oc=0; oc<OC; oc++)
        {
            for(int ic=0; ic<IC; ic++) 
            {
                Complex[] C = fft_corr(A[n][ic], B[oc][ic]);
                
                int index = 0;
                for(int oh=0; oh<OH; oh++)
                for(int ow=0; ow<OW; ow++)
                    Y[n][oh][ow][oc] += C[index++].real();
            }
        }
    }
    
    public static Complex[] to1D(Complex[][][][] X)
    {
        int dim0 = X.length;
        int dim1 = X[0].length;
        int dim2 = X[1].length;
        int dim3 = X[2].length;
        
        Complex[] A = new Complex[dim0 * dim1 * dim2 * dim3];
        int index = 0;
        for(int d0=0; d0<dim0; d0++)
        for(int d1=0; d1<dim1; d1++)
        for(int d2=0; d2<dim2; d2++)
        for(int d3=0; d3<dim3; d3++) 
            A[index++] = X[d0][d1][d2][d3];
        return A;
    }
}
