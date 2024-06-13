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
import z.dragon.engine.Tensor;
import z.dragon.engine.memp.Mempool;

/**
 *
 * @author Gilgamesh
 */
public class AT5 
{
    static { alpha.home("C:\\Users\\Gilgamesh\\Desktop\\Dragon-alpha-v1.1"); }
    static Mempool memp = alpha.engine.memp1(alpha.MEM_1GB * 8);
    static Engine eg = alpha.engine.cuda_float32(0, memp, alpha.MEM_1MB * 1024);
    
    static String path = "C:\\Users\\Gilgamesh\\Desktop\\sj22.jpg";
    
    public static void test1() throws IOException {
        BufferedImage img = cv.imread(path);
        int IH = img.getHeight(), IW = img.getWidth();
        Tensor X = eg.img.pixels(img);
        
        img = cv.BGR(eg.img.linear(false, 1.0f, X, 44)); cv.imshow(img, "x2");
        System.out.println(cv.brief(img));
        
        img = cv.BGR(eg.img.linear(false, 1.5f, X, -43)); cv.imshow(img, "x3");
        System.out.println(cv.brief(img));
    }
    
    public static void test2() throws IOException {
        BufferedImage img = cv.imread(path);
        int IH = img.getHeight(), IW = img.getWidth();
        Tensor X = eg.img.pixels(img);
        
        img = cv.BGR(eg.img.exp(false, 0.05f, X, 0.0f, -64)); cv.imshow(img, "x2");
        System.out.println(cv.brief(img));
        
        img = cv.BGR(eg.img.exp(false, 0.1f, X, 0.0f, -53)); cv.imshow(img, "x3");
        System.out.println(cv.brief(img));
    }
    
    public static void test3() throws IOException {
        BufferedImage img = cv.imread(path);
        int IH = img.getHeight(), IW = img.getWidth();
        Tensor X = eg.img.pixels(img);
        
        img = cv.BGR(eg.img.log(false, 114.0f, 0.1f, X, 1.0f)); cv.imshow(img, "x2");
        System.out.println(cv.brief(img));
        
        img = cv.BGR(eg.img.log(false, 44.0f, 1f, X, 1.0f)); cv.imshow(img, "x3");
        System.out.println(cv.brief(img));
    }
    
     public static void test4() throws IOException {
        BufferedImage img = cv.imread(path); cv.imshow(img, "x2");
        
        Tensor X = eg.img.pixels(img);
        X = eg.img.pixel_to_dtype(true, X);
        X = eg.img.dtype_to_pixel(true, X);
        
        img = cv.BGR(X); cv.imshow(img, "x2");
        System.out.println(cv.brief(img));
    }
    
    public static void main(String[] args)
    {
        try
        {
//            test1();
//            test2();
//            test3();
            test4();
        }
        catch(IOException e) {
            e.printStackTrace();
        }
    }
}
