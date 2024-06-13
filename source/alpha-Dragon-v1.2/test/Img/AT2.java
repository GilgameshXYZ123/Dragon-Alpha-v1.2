/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package Img;
import java.awt.image.BufferedImage;
import java.io.IOException;
import static z.dragon.alpha.Alpha.alpha;
import static z.dragon.common.DragonCV.cv;
import z.dragon.engine.Engine;
import z.dragon.engine.ImageEngine.ImageAffiner;
import z.dragon.engine.Tensor;
import z.dragon.engine.memp.Mempool;

/**
 *
 * @author Gilgamesh
 */
public class AT2 
{
    static { alpha.home("C:\\Users\\Gilgamesh\\Desktop\\Dragon-alpha-v1.1"); }
    static Mempool memp = alpha.engine.memp1(alpha.MEM_1GB * 8);
    static Engine eg = alpha.engine.cuda_float32(0, memp, alpha.MEM_1MB * 1024);
    
    static String path = "C:\\Users\\Gilgamesh\\Desktop\\sj22.jpg";
    
    //<editor-fold defaultstate="collapsed" desc="operations">
    public static float[] indentity() { return new float[] { 1, 0, 0, 0, 1, 0 }; }
    
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
        float M00 = m00 + shx*m10, M10 = shy*m00 + m10;
        float M01 = m01 + shx*m11, M11 = shy*m01 + m11;
        float M02 = m02 + shx*m12, M12 = shy*m02 + m12;
        return new float[] { M00, M01, M02, M10, M11, M12 };
    }
    
    //[cosx, -sinx, 0]   [m00, m01, m02]
    //[sinx,  cosx, 0] * [m10, m11, m12]
    //[   0,     0, 1]   [  0,   0,   1]
    public static float[] rotate(float[] M, float x) {
        float m00 = M[0], m01 = M[1], m02 = M[2];
        float m10 = M[3], m11 = M[4], m12 = M[5];
        double sinx = Math.sin(x), cosx = Math.cos(x);
        
        float M00 = (float) (cosx*m00 - sinx*m10);
        float M01 = (float) (cosx*m01 - sinx*m11);
        float M02 = (float) (cosx*m02 - sinx*m12);
        float M10 = (float) (sinx*m00 + cosx*m10);
        float M11 = (float) (sinx*m01 + cosx*m11);
        float M12 = (float) (sinx*m02 + cosx*m12);

        return new float[] { M00, M01, M02, M10, M11, M12 };
    }
    
    //[1, 0, -tx]   [cosx, -sinx, 0]   [1, 0, tx]
    //[0, 1, -ty] * [sinx,  cosx, 0] * [0, 1, ty]
    //[0, 0,   1]   [   0,     0, 1]   [0, 0,  1]
    public static float[] rotate(float[] M, float x, float cy, float cx) {
        M = translate(M, cy, cx);
        M = rotate(M, x);
        M = translate(M, -cy, -cx);
        return M;
    }
    
    //[1, 0, tx]   [cosx, -sinx, 0]   [1, 0, -tx]
    //[0, 1, ty] * [sinx,  cosx, 0] * [0, 1, -ty]
    //[0, 0,  1]   [   0,     0, 1]   [0, 0,   1]
    public static float[] rotate2(float[] M, float x, float cy, float cx) {
        float m00 = M[0], m01 = M[1], m02 = M[2];
        float m10 = M[3], m11 = M[4], m12 = M[5];
        
        double sinx = Math.sin(x), cosx = Math.cos(x);
        double a00 = cosx, a01 = -sinx, a02 = cosx*cx - sinx*cy - cx;
        double a10 = sinx, a11 =  cosx, a12 = sinx*cx + cosx*cy - cy;
        
        float M00 = (float) (a00*m00 + a01*m10);
        float M01 = (float) (a00*m01 + a01*m11);
        float M02 = (float) (a00*m02 + a01*m12 + a02);
        
        float M10 = (float) (a10*m00 + a11*m10);
        float M11 = (float) (a10*m01 + a11*m11);
        float M12 = (float) (a10*m02 + a11*m12 + a12);
        
        return new float[] { M00, M01, M02, M10, M11, M12 };
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="auxilary">
    public static float min(float... x) {
        float min = x[0];
        for(int i=1; i<x.length; i++) if(x[i] < min) min = x[i]; 
        return min;
    }
    
    public static float max(float... x) {
        float max = x[0];
        for(int i=1; i<x.length; i++) if(x[i] > max) max = x[i]; 
        return max;
    }
    
    public static float[] revert(float[] M) {//find the reverse of M
        float m00 = M[0], m01 = M[1], m02 = M[2];
        float m10 = M[3], m11 = M[4], m12 = M[5];
        
        float det = m00*m11 - m01*m10;
        float r00 =  m11 / det, r01 = -m01 / det, r02 = (m01*m12 - m11*m02)/det;
        float r10 = -m10 / det, r11 =  m00 / det, r12 = (m10*m02 - m00*m12)/det;
        return new float[] { r00, r01, r02, r10, r11, r12 };
    }
    //</editor-fold>
    
    public static Object[] affine(byte[] X, int IH, int IW, int C, float[] M)  
    {
        float m00 = M[0], m01 = M[1], m02 = M[2];//u = m00*x + m01*y + m02
        float m10 = M[3], m11 = M[4], m12 = M[5];//v = m10*x + m11*y + m12
        
        //(0, 0) -> (oh0, ow0), (IH, IW) -> (oh3, ow3)
        float ow0 = m02, ow3 = m00*IW + m01*IH + m02;
        float oh0 = m12, oh3 = m10*IW + m11*IH + m12;
        
        //(0, IW) -> (oh1, ow1), (IH, 0) -> (oh2, ow2)
        float ow1 = m00*IW + m02, ow2 = m01*IH + m02;
        float oh1 = m10*IW + m12, oh2 = m11*IH + m12;
        
        int OH = (int) Math.ceil(max(oh0, oh1, oh2, oh3));
        int OW = (int) Math.ceil(max(ow0, ow1, ow2, ow3));
        
        float[] R = revert(M);
        float r00 = R[0], r01 = R[1], r02 = R[2];//x = r00*u + r01*v + r02
        float r10 = R[3], r11 = R[4], r12 = R[5];//y = r10*u + r11*v + r12
        
        System.out.println(r00 + ", " + r01 + ", " + r02);
        System.out.println(r10 + ", " + r11 + ", " + r12);
       
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
    
    public static void test1() throws IOException {
        BufferedImage img = cv.imread(path); //cv.imshow(img, "original");
        int IH = img.getHeight();
        int IW = img.getWidth();
        byte[] pixel = cv.pixel(img);
        System.out.println(cv.brief(img));
        
        float[] M = indentity();
        
        M = translate(M, 100, 100);
        M = scale(M, 0.3f, 0.3f);
//        M = shear(M, 0.3f, 0.3f); 
        M = rotate2(M, (float) (30 * Math.PI / 180), 40, 40);
        
        Object[] result = affine(pixel, IH, IW, 3, M);
        int OH = (int) result[0];
        int OW = (int) result[1];
        pixel = (byte[]) result[2];
        BufferedImage img2 = cv.BGR(pixel, OH,OW); cv.imshow(img2);
        System.out.println(cv.brief(img2));
    }
    
    public static void test2() throws IOException {
        BufferedImage img = cv.imread(path); //cv.imshow(img, "original");
        int IH = img.getHeight();
        int IW = img.getWidth();
        byte[] pixel = cv.pixel(img);
        System.out.println(cv.brief(img));
        
        Tensor X = eg.tensor_int8(pixel, IH, IW, 3);
        ImageAffiner af = eg.img.affine()
                .translate(100, 100)
                .scale(0.3f, 0.3f)
//                .shear(0.3f, 0.3f)
                .rotate((float) (30 * Math.PI / 180), 40, 40);
        X  = af.transform(X, true);
        System.out.println(af.invert());
        
        
        pixel = X.value_int8();
        int OH = X.dim(0), OW = X.dim(1);
        BufferedImage img2 = cv.BGR(pixel, OH,OW); cv.imshow(img2);
        System.out.println(cv.brief(img2));
    }
    
    public static void test3() throws IOException {
        BufferedImage img = cv.imread(path);
        
        Tensor X = eg.img.pixels(img);
        X.img().scale(true, 0.5f, 0.5f);
        img = cv.BGR(X); cv.imshow(img, "original");
        
        X.img().linear(true, 2f, 0).pixel();
        img = cv.BGR(X); cv.imshow(img, "img2");
    }
    
    public static void main(String[] args)
    {
        try 
        {
            test1();
            test2();
//             test3();
        }
        catch(Exception e) {
            e.printStackTrace();
        }
    }
}
