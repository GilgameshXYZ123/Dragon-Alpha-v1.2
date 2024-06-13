/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package Img;

import static Img.Demo.test1;
import java.awt.image.BufferedImage;
import java.io.IOException;
import static z.dragon.alpha.Alpha.alpha;
import static z.dragon.common.DragonCV.cv;
import z.dragon.engine.Engine;
import z.dragon.engine.memp.Mempool;
import z.util.math.vector.Vector;

/**
 *
 * @author Gilgamesh
 */
public class AT 
{
    static { alpha.home("C:\\Users\\Gilgamesh\\Desktop\\Dragon-alpha-v1.2"); }
    static Mempool memp = alpha.engine.memp1(alpha.MEM_1GB * 8);
    static Engine eg = alpha.engine.cuda_float32(0, memp, alpha.MEM_1MB * 1024);
    
    static String path = "C:\\Users\\Gilgamesh\\Desktop\\sj22.jpg";
    
    public static float[] revert(float[] M) {//find the reverse of M
        float m00 = M[0], m01 = M[1], m02 = M[2];
        float m10 = M[3], m11 = M[4], m12 = M[5];
        
        float det = m00*m11 - m01*m10;
        float r00 =  m11 / det, r01 = -m01 / det, r02 = (m01*m12 - m11*m02)/det;
        float r10 = -m10 / det, r11 =  m00 / det, r12 = (m10*m02 - m00*m12)/det;
        return new float[] { r00, r01, r02, r10, r11, r12 };
    }
    
    public static float min(float... x) {
        float max = x[0];
        for(int i=1; i<x.length; i++) if(x[i] > max) max = x[i]; 
        return max;
    }
    
    public static float max(float... x) {
        float max = x[0];
        for(int i=1; i<x.length; i++) if(x[i] > max) max = x[i]; 
        return max;
    }
    
    public static int[] affine_out_size(int IH, int IW, float[] M) {
        float m00 = M[0], m01 = M[1], m02 = M[2];//u = m00*x + m01*y + m02
        float m10 = M[3], m11 = M[4], m12 = M[5];//v = m10*x + m11*y + m12
        
        //(0, 0) -> (oh0, ow0)
        float ow0 = m02;
        float oh0 = m12;
        
        //(0, IW) -> (oh1, ow1)
        float ow1 = m00*IW + m02;
        float oh1 = m10*IW + m12;
        
        //(IH, 0) -> (oh2, ow2)
        float ow2 = m01*IH + m02;
        float oh2 = m11*IH + m12;
        
        //(IH, IW) ->(oh3, ow3)
        float ow3 = m00*IW + m01*IH + m02;
        float oh3 = m10*IW + m11*IH + m12;
        
        int OW = (int) (max(ow0, ow1, ow2, ow3));
        int OH = (int) (max(oh0, oh1, oh2, oh3));
        
        return new int[] { OH, OW };
    }
    
    
    public static Object[] affine(byte[] X, int IH, int IW, int C, float[] M)  {
        float[] R = revert(M);
        float r00 = R[0], r01 = R[1], r02 = R[2];//x = r00*u + r01*v + r02
        float r10 = R[3], r11 = R[4], r12 = R[3];//y = r10*u + r11*v + r12
        
        int[] out_size = affine_out_size(IH, IW, M);
        int OH = out_size[0], OW = out_size[1];
        byte[] Y = new byte[OH * OW * C];
        
        for(int oh=0; oh<OH; oh++)//v = oh: OH -> y = ih
        for(int ow=0; ow<OW; ow++)//u = ow: OW -> x = iw
        {
            int u = ow, v = oh;
            int x = (int) (r00*u + r01*v + r02);
            int y = (int) (r10*u + r11*v + r12);
            
            int iw = x, ih = y;
            boolean in_range = (ih >= 0) && (ih < IH) && (iw >= 0) && (iw < IW);
            for(int c=0; c<C; c++) {
                int yindex = (oh*OW + ow)*C + c;
                int xindex = (ih*IW + iw)*C + c;
                Y[yindex] = (in_range? X[xindex] : 0);
            }
        }
        return new Object[]{ OH, OW, Y };
    }
    
    public static float[] indentity() {
        return new float[] { 1, 0, 0, 0, 1, 0 };
    }
    
    //[1, 0, tx]   [m00, m01, m02]  
    //[0, 1, ty] * [m10, m11, m12]
    //[0, 0,  1]   [  0,   0,   1]
    public static float[] translate(float[] M, float ty, float tx) {//passed
        float m00 = M[0], m01 = M[1], m02 = M[2];
        float m10 = M[3], m11 = M[4], m12 = M[5];
        m02 = m02 + tx;
        m12 = m12 + ty;
        return new float[] { m00, m01, m02, m10, m11, m12 };
    }
    
    //[sx,  0, 0]   [m00, m01, m02]
    //[ 0, sy, 0] * [m10, m11, m12]
    //[ 0,  0, 1]   [  0,   0,   1]
    public static float[] scale(float[] M, float sy, float sx) {//passed
        float m00 = M[0], m01 = M[1], m02 = M[2];
        float m10 = M[3], m11 = M[4], m12 = M[5];
        m00 = m00*sx; m01 = m01*sx; m02 = m02*sx;
        m10 = m10*sy; m11 = m11*sy; m12 = m12*sy;
        return new float[] { m00, m01, m02, m10, m11, m12 };
    }
    
    //[  1, shx, 0]   [m00, m01, m02]
    //[shy,   1, 0] * [m10, m11, m12]
    //[  0,   0, 1]   [  0,   0,   1]
    public static float[] shear(float[] M, float shy, float shx) {//passed
        float m00 = M[0], m01 = M[1], m02 = M[2];
        float m10 = M[3], m11 = M[4], m12 = M[5];
        m00 = m00 + shx*m10; m10 = shy*m00 + m10;
        m01 = m01 + shx*m11; m11 = shy*m01 + m11;
        m02 = m02 + shx*m12; m12 = shy*m02 + m12;
        return new float[] { m00, m01, m02, m10, m11, m12 };
    }
    
    //[1,  0,      0]   [m00, m01, m02]
    //[0, -1, height] * [m10, m11, m12]
    //[0,  0,      1]   [  0,   0,   1]
    public static float[] reflect(float[] M, int IH, int IW) {
        int[] out_size = affine_out_size(IH, IW, M);
        IH = out_size[0]; IW = out_size[1];
        System.out.println(IH);
        
        float m00 = M[0], m01 = M[1], m02 = M[2];
        float m10 = M[3], m11 = M[4], m12 = M[5];
        m10 = -m10;
        m11 = -m11;
        m12 = -m12 + IH;
        return new float[] { m00, m01, m02, m10, m11, m12 };
    }
    
    //[cosx, -sinx, 0]   [m00, m01, m02]
    //[sinx,  cosx, 0] * [m10, m11, m12]
    //[   0,     0, 1]   [  0,   0,   1]
    public static float[] rotate(float[] M, float x) {
        float m00 = M[0], m01 = M[1], m02 = M[2];
        float m10 = M[3], m11 = M[4], m12 = M[5];
        
        double sinx = Math.sin(x);
        double cosx = Math.cos(x);

//        m00 = (float) (cosx*m00 - sinx*m10);
//        m01 = (float) (cosx*m01 - sinx*m11);
//        m02 = (float) (cosx*m02 - sinx*m12);
//        m10 = (float) (sinx*m00 + cosx*m10);
//        m11 = (float) (sinx*m01 + cosx*m11);
//        m12 = (float) (sinx*m02 + cosx*m12);
        
        m00 = (float) (cosx*m00 + sinx*m01);
        m10 = (float) (cosx*m10 + sinx*m11);
        m01 = (float) (-sinx*m00 + cosx*m01);
        m11 = (float) (-sinx*m10 + cosx*m11);
        
        return new float[] { m00, m01, m02, m10, m11, m12 };
    }
    
    public static void test1() throws Exception {
        BufferedImage img = cv.imread(path); //cv.imshow(img, "original");
        int IH = img.getHeight();
        int IW = img.getWidth();
        byte[] pixel = cv.pixel(img);
        System.out.println(cv.brief(img));
        
        float[] M = indentity();
        
//        M = translate(M, 100, 100);
//        M = scale(M, 0.5f, 0.5f);
//        M = rotate(M, (float) (0.3*Math.PI));
//        M = shear(M, 0.3f, 0.3f); 
        M = reflect(M, IH, IW);
                
        Object[] result = affine(pixel, IH, IW, 3, M);
        int OH = (int) result[0];
        int OW = (int) result[1];
        pixel = (byte[]) result[2];
        BufferedImage img2 = cv.BGR(pixel, OH,OW); cv.imshow(img2);
        System.out.println(cv.brief(img2));
    }
    
    public static void test2() throws Exception {
        BufferedImage img = cv.imread(path); //cv.imshow(img, "original");
        img = cv.affine()
//                .show_move(100, 100)
//                .scale(0.5, 0.5)
//                .shear(0.3, 0.3)
                .rotate((float) (50 * Math.PI / 180))
                .transform(img);
        
        System.out.println(cv.brief(img)); 
        cv.imshow(img);
    }
    
    public static void main(String[] args)
    {
        try 
        {
            double[] A = new double[]{ 2032.1, 1152.1, 1016.3 ,1024.4,1024.4,1016.6,1016.6, 784.8, 509, 513.3 };
            System.out.println(Vector.average(A));
            
//            test2();
        }
        catch(Exception e) {
            e.printStackTrace();
        }
    }
}
