/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.util.macler.linreg;

import z.util.math.vector.Matrix;
import z.util.math.vector.Vector;

/**
 *<pre>
 * Linear Regression.
 * 1.least square method(Matrix representation and Geometric meaning)
 * =>L1->lasso     =>L2->Ridage 
 * 
 * 2.DataSet {(xi,yi)|xi belongs to X, y belongs to Y}
 * find a hyper line: y=Wt*x+b.
 * let w0*x0=b0, w0=b0, x0=1
 * W=(w0, w1,...wp)t
 * x=(x0, x1,...xp)t
 * X=(x1, x2,...xn)
 * Y=(y1, y2,...yn)
 *
 * 3.
 * 
 * </pre>
 * @author dell
 */
public final class LinearRegression 
{
    private LinearRegression() {}
    
    private static double[][] expendW(double[][] x)
    {
        int width=x[0].length;
        double[][] ex=new double[x.length+1][width];
        
        for(int j=0;j<width;j++) ex[0][j]=1;
        for(int i=1,j;i<ex.length;i++)
        for(j=0;j<ex[i].length;j++) ex[i][j]=x[i-1][j];
        return ex;
    }
    /**
     * <pre>
     * With out regulairzation.
     * When programming, we regaed:
     * (1)each column of Matrix X as a vector
     * (2)double[] W as a field vector
     * (3)double[] Y as a line vectir
     * (4)b=w0.
     *  L(W)=(Wt*X-Y)*(Wt*X-Y)t
     *     =Wt*X*Xt*W-2Y*Xt*W+Y*Yt
     * 
     * dL/dx=2Wt*X*Xt-2Y*Xt=0
     * Wt*X*Xt=Y*Xt
     * Wt=Y*Xt * inversed(X*Xt)
     * Wt=Y*Xt * inversed(Xt)*inversed(X)
     * Wt=Y * inerversed(X)
     * </pre>
     * @param x
     * @param y
     * @return 
     */
    private static double[] findW(double[][] x, double[] y)
    {
        //Wt=Y*Xt*(X*Xt)-1
        x=expendW(x);//x0=1, x=[p+1, n]
        double[] y_xt=Vector.multiply(y, Matrix.transpose(x));//y_xt=(1,n)*(n,p+1）=(1,p+1)
        double[][] xtx_i=Matrix.inverse(Matrix.multiplyT(x, x));//(p+1, n)*(n, p+1)=(p+1,p+1)
        return Vector.multiply(xtx_i, y_xt);//(p+1, p+1)*(p+1, 1)=(p+1, 1) b=w0
    }
    /**
     *<pre>
     * L2->Ridege. 
     * L(W)=sum[i,0,n] (Wt*X-Y)**2 + S*||W||**2; 
     *     =sum[i,0,n](Wt*X-Y)**2 + S*Wt*W 
     *     =(Wt*X-Y)*(Wt*X-Y)t + S*Wt*W 
     *     =Wt*X*Xt - 2Y*Xt*W + Y*Yt + S*Wt*W 
     * 
     * dL/dx=2Wt*X*Xt - 2Y*Xt + 2S*W=0 
     * => Wt*X*Xt - Y*Xt + S*Wt=0
     *    Wt(X*Xt+S)=Y*Xt
     *    Wt= inversed(X*Xt+S) + Y*Xt
     * </pre>
     * @param x
     * @param y
     * @return
     */
    private static double[] findWRidge(double[][] x ,double[] y, double s)
    {
        x=expendW(x);
        double[] y_xt=Vector.multiply(y, Matrix.transpose(x));
        double[][] xtx_i=Matrix.multiplyT(x, x);
        Matrix.add(xtx_i, xtx_i, s);
        return Vector.multiply(Matrix.inverse(xtx_i), y_xt);
    }
}
